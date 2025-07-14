import os
import glob
import pandas as pd
import re
import numpy as np
import streamlit as st
from unidecode import unidecode
from fuzzywuzzy import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(
    page_title="Product Matching Across Stores",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'model' not in st.session_state:
    st.session_state.model = None

def load_csv_datasets(directory):
    pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(pattern)
    datasets = {}
    for filepath in csv_files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        df = pd.read_csv(filepath)
        datasets[name] = df
    return datasets

def normalize_product_title(title):
    if pd.isna(title):
        return ""
    
    # Convert to string and decode unicode characters
    title = str(title)
    title = unidecode(title)
    
    # Convert to lowercase
    title = title.lower()
    
    # Remove extra whitespace and normalize spaces
    title = re.sub(r'\s+', ' ', title.strip())
    
    # Remove special characters but keep alphanumeric, spaces, and common punctuation
    title = re.sub(r'[^\w\s\-\.\,\&\(\)]', '', title)
    
    # Standardize common abbreviations and units
    replacements = {
        r'\bpk\b': 'pack',
        r'\bpkt\b': 'pack', 
        r'\bmg\b': 'milligram',
        r'\bml\b': 'millilitre',
        r'\bl\b': 'litre',
        r'\bg\b': 'gram',
        r'\bkg\b': 'kilogram',
        r'\boz\b': 'ounce',
        r'\blb\b': 'pound',
        # Remove the problematic X replacements that interfere with size extraction
        # r'\bx\s*(\d+)': r'pack of \1',
        # r'(\d+)\s*x\s*(\d+)': r'\1 pack of \2',
        r'\&': 'and',
        r'\bw\/': 'with',
        r'\bw\b': 'with'
    }
    
    for pattern, replacement in replacements.items():
        title = re.sub(pattern, replacement, title)
    
    # Remove extra spaces again after replacements
    title = re.sub(r'\s+', ' ', title.strip())
    
    return title

def extract_size_and_clean_title(title):
    """
    Extract size info (including multi-pack) and return cleaned title without size.
    """
    if pd.isna(title) or not title.strip():
        return "", title
    
    text = title.strip()
    
    # Map word numbers to digits
    word_to_num = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12'
    }
    
    extracted_size = ""
    
    # Define patterns with their handlers (order matters - more specific first)
    patterns = [
        # Word pack of size: "six pack of 375ml"
        (r'\b(' + '|'.join(word_to_num.keys()) + r')\s+pack(?:s)?\s+of\s+(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)',
         lambda m: f"{m.group(2)}{m.group(3)}x{word_to_num[m.group(1)]}"),
        
        # Numeric pack of size: "12 pack of 330ml"
        (r'\b(\d+)\s+pack(?:s)?\s+of\s+(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)',
         lambda m: f"{m.group(2)}{m.group(3)}x{m.group(1)}"),
        
        # Size x count: "375ml x 6", "375ml x6", "375ml X 24"
        (r'\b(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)\s*[xX×]\s*(\d+)',
         lambda m: f"{m.group(1)}{m.group(2)}x{m.group(3)}"),
        
        # Count x size: "6 x 375ml", "24 X 375ml"
        (r'\b(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)',
         lambda m: f"{m.group(2)}{m.group(3)}x{m.group(1)}"),
        
        # Size with pack mention (various formats): "375ml 24 pack", "375ml X 24 Pack", "375ml multipack 24"
        (r'\b(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)\s*(?:x|X|×)?\s*(?:multipack)?\s*(?:cans?)?\s*(\d+)\s*pack',
         lambda m: f"{m.group(1)}{m.group(2)}x{m.group(3)}"),
        
        # Pack with size mention: "24 pack 375ml", "24 Pack X 375ml", "24 multipack 375ml"
        (r'\b(\d+)\s*(?:pack|multipack)\s*(?:x|X|×)?\s*(?:cans?)?\s*(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)',
         lambda m: f"{m.group(2)}{m.group(3)}x{m.group(1)}"),
        
        # Alternative multipack format: "Multipack Cans 375ml X 24 Pack"
        (r'\bmultipack\s+cans?\s+(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)\s*[xX×]\s*(\d+)\s*pack',
         lambda m: f"{m.group(1)}{m.group(2)}x{m.group(3)}"),
        
        # Another alternative: "Cans 375ml X 24 Pack"
        (r'\bcans?\s+(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)\s*[xX×]\s*(\d+)\s*pack',
         lambda m: f"{m.group(1)}{m.group(2)}x{m.group(3)}"),
        
        # Simple size: "375ml", "1.25l", "45g" (must come after multipack patterns)
        (r'\b(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg|ounce|oz|pound|lb)',
         lambda m: f"{m.group(1)}{m.group(2)}"),
        
        # Pack only: "6 pack", "12pack"
        (r'\b(\d+)\s*pack\b',
         lambda m: f"{m.group(1)}-pack"),
        
        # Word pack only: "six pack"
        (r'\b(' + '|'.join(word_to_num.keys()) + r')\s+pack\b',
         lambda m: f"{word_to_num[m.group(1)]}-pack"),
    ]
    
    # Find and extract the first matching pattern
    for pattern, handler in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_size = handler(match)
            # Remove the matched pattern from the title
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            break
    
    # Clean up the title: remove extra spaces and common connecting words
    cleaned_title = re.sub(r'\s+', ' ', text).strip()
    cleaned_title = re.sub(r'\b(of|with|in)\s*$', '', cleaned_title, flags=re.IGNORECASE).strip()
    
    return extracted_size, cleaned_title

def normalize_size_field(size):
    """
    Normalize size field to consistent format.
    """
    if not isinstance(size, str) or not size.strip():
        return ""
    
    s = size.strip().lower()
    
    # Handle multi-pack format: "375mlx6"
    multipack_match = re.match(r'(\d+(?:\.\d+)?)(ml|l|g|kg)x(\d+)', s)
    if multipack_match:
        val, unit, count = multipack_match.groups()
        val = float(val)
        # Convert to standard units
        if unit == 'kg':
            val = int(val * 1000)
            unit = 'g'
        elif unit == 'l':
            val = int(val * 1000)
            unit = 'ml'
        else:
            val = int(val) if val.is_integer() else val
        return f"{val}{unit}x{count}"
    
    # Handle pack patterns
    pack_match = re.match(r'pack of\s*(\d+)', s)
    if pack_match:
        return f"{pack_match.group(1)}-pack"
    
    pack_match = re.match(r'(\d+)\s*-?\s*pack', s)
    if pack_match:
        return f"{pack_match.group(1)}-pack"
    
    # Handle weight/volume patterns
    size_match = re.match(r'([\d\.]+)\s*(kg|g|l|ml|oz|lb)\b', s)
    if size_match:
        val, unit = float(size_match.group(1)), size_match.group(2)
        
        # Convert to standard units
        if unit == 'kg':
            return f"{int(val * 1000)}g"
        elif unit == 'l':
            return f"{int(val * 1000)}ml"
        elif unit in ['oz', 'lb']:
            # Keep imperial units as-is for now
            val = int(val) if val.is_integer() else val
            return f"{val}{unit}"
        else:  # g or ml
            val = int(val) if val.is_integer() else val
            return f"{val}{unit}"
    
    # Fallback: collapse whitespace and return as-is
    return re.sub(r'\s+', '_', s)

def normalize_brand_name_fuzzy(brand):
    """
    Normalize brand names using fuzzywuzzy for fuzzy matching.
    """
    if pd.isna(brand) or not brand.strip():
        return ""
    
    # Clean the input brand
    cleaned_brand = str(brand).strip().lower()
    cleaned_brand = re.sub(r'[^\w\s]', '', cleaned_brand)
    cleaned_brand = re.sub(r'\s+', ' ', cleaned_brand).strip()
    
    if not cleaned_brand:
        return ""
    
    # Standard brand names (canonical forms)
    standard_brands = [
        'Coca-Cola', 'Pepsi', 'Sprite', 'Fanta', 'Mountain Dew', 'Dr Pepper',
        'Schweppes', 'Red Bull', 'Monster Energy', '7 Up',
        'Cadbury', 'Nestlé', 'Mars', 'Snickers', 'KitKat', 'Twix', 
        'M&Ms', 'Kinder', 'Aero', 'Milky Way', 'Bounty', 'Toblerone',
        'Ferrero Rocher', 'Lindt', 'Godiva', 'Hershey',
        'Woolworths', 'Coles', 'IGA', 'Aldi', 'Macro', 'Woolworths Select',
        'Coles Smart Buy', 'Black & Gold', 'Home Brand'
    ]
    
    # Create lowercase versions for matching
    standard_brands_lower = [brand.lower() for brand in standard_brands]
    
    # Try exact match first
    if cleaned_brand in standard_brands_lower:
        idx = standard_brands_lower.index(cleaned_brand)
        return standard_brands[idx]
    
    # Use fuzzywuzzy to find the best match
    best_match, score = process.extractOne(
        cleaned_brand, 
        standard_brands_lower,
        scorer=fuzz.ratio
    )
    
    # Set threshold for fuzzy matching
    threshold = 75  
    
    if score >= threshold:
        # Return the canonical form
        idx = standard_brands_lower.index(best_match)
        return standard_brands[idx]
    
    # If no good match found, try partial matching for compound names
    best_partial_match, partial_score = process.extractOne(
        cleaned_brand,
        standard_brands_lower,
        scorer=fuzz.partial_ratio
    )
    
    partial_threshold = 85  # Higher threshold for partial matching
    if partial_score >= partial_threshold:
        idx = standard_brands_lower.index(best_partial_match)
        return standard_brands[idx]
    
    # If still no match, capitalize properly and return
    words = cleaned_brand.split()
    capitalized_words = []
    
    for word in words:
        if len(word) <= 2:
            # Keep short words uppercase (like initials)
            capitalized_words.append(word.upper())
        else:
            # Capitalize first letter of longer words
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)

def create_product_signature(row):
    """
    Creating a text signature for product matching using only the cleaned product title.
    This is used for the final similarity comparison after strict filtering.
    """
    # Only use cleaned product title for similarity comparison
    if pd.notna(row.get('productTitle', '')) and row.get('productTitle', '').strip():
        return str(row['productTitle']).strip().lower()
    return ""

def normalize_brand_for_matching(brand):
    """
    Normalize brand name specifically for exact matching.
    More aggressive normalization for brand comparison.
    """
    if pd.isna(brand) or not brand.strip():
        return ""
    
    # Clean and normalize
    cleaned = str(brand).strip().lower()
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Remove common suffixes and prefixes
    cleaned = re.sub(r'\b(co|company|ltd|limited|inc|corp|corporation)\b', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Handle common variations
    brand_variations = {
        'coca cola': 'coca-cola',
        'cocacola': 'coca-cola',
        'coke': 'coca-cola',
        'pepsi cola': 'pepsi',
        'mountain dew': 'mountain-dew',
        'dr pepper': 'dr-pepper',
        'seven up': '7-up',
        '7up': '7-up'
    }
    
    return brand_variations.get(cleaned, cleaned)

def normalize_size_for_matching(size):
    """
    Normalize size field for exact matching.
    """
    if pd.isna(size) or not size.strip():
        return ""
    
    size_str = str(size).strip().lower()
    
    # Handle different size formats and convert to standard format
    # Convert liters to ml for consistency
    size_str = re.sub(r'(\d+(?:\.\d+)?)\s*l\b', lambda m: f"{int(float(m.group(1)) * 1000)}ml", size_str)
    size_str = re.sub(r'(\d+(?:\.\d+)?)\s*litre', lambda m: f"{int(float(m.group(1)) * 1000)}ml", size_str)
    
    # Convert kg to g for consistency
    size_str = re.sub(r'(\d+(?:\.\d+)?)\s*kg\b', lambda m: f"{int(float(m.group(1)) * 1000)}g", size_str)
    size_str = re.sub(r'(\d+(?:\.\d+)?)\s*kilogram', lambda m: f"{int(float(m.group(1)) * 1000)}g", size_str)
    
    # Standardize pack notation
    size_str = re.sub(r'(\d+)\s*-?\s*pack', r'\1-pack', size_str)
    
    # Remove extra spaces
    size_str = re.sub(r'\s+', '', size_str)
    
    return size_str

def brands_match_fuzzy(brand1, brand2, threshold=85):
    """
    Check if two brands match using fuzzy string matching with partial matching.
    """
    if not brand1 or not brand2:
        return False
    
    # Normalize both brands
    norm_brand1 = normalize_brand_for_matching(brand1)
    norm_brand2 = normalize_brand_for_matching(brand2)
    
    # Exact match first
    if norm_brand1 == norm_brand2:
        return True
    
    # Partial ratio matching (good for "coca" vs "coca-cola")
    partial_similarity = fuzz.partial_ratio(norm_brand1, norm_brand2)
    if partial_similarity >= threshold:
        return True
    
    # Regular fuzzy match
    similarity = fuzz.ratio(norm_brand1, norm_brand2)
    return similarity >= threshold

@st.cache_resource
def load_sentence_transformer():
    """Cache the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def find_identical_products_across_stores(input_row, dataframe):
    """
    Find identical products across different stores using a hybrid approach:
    1. Strict filtering based on brand and size
    2. Similarity scoring on product title only
    3. High threshold for final matches
    """
    model = load_sentence_transformer()
    
    # Get input product details
    input_store = input_row.get('shop', 'Unknown')
    input_brand = input_row.get('brandName', '')
    input_size = input_row.get('size', '')
    input_title = input_row.get('productTitle', '')
    
    # Normalize input attributes for matching
    input_brand_norm = normalize_brand_for_matching(input_brand)
    input_size_norm = normalize_size_for_matching(input_size)
    input_signature = create_product_signature(input_row)
    
    if not input_signature or not input_brand_norm or not input_size_norm:
        return {'most_identical': [], 'closely_related': [], 'total_matches': 0}
    
    # Filter out products from the same store
    other_stores_df = dataframe[dataframe['shop'] != input_store].copy()
    
    if len(other_stores_df) == 0:
        return {'most_identical': [], 'closely_related': [], 'total_matches': 0}
    
    # Step 1: Strict Filtering - Brand and Size must match
    print(f"Starting with {len(other_stores_df)} products from other stores")
    
    # Filter by brand (fuzzy matching)
    brand_candidates = []
    for idx, row in other_stores_df.iterrows():
        candidate_brand = row.get('brandName', '')
        if brands_match_fuzzy(input_brand, candidate_brand):
            brand_candidates.append(idx)
    
    if not brand_candidates:
        print("No brand matches found")
        return {'most_identical': [], 'closely_related': [], 'total_matches': 0}
    
    brand_filtered_df = other_stores_df.loc[brand_candidates].copy()
    print(f"After brand filtering: {len(brand_filtered_df)} products")
    
    # Filter by size (exact matching)
    size_candidates = []
    for idx, row in brand_filtered_df.iterrows():
        candidate_size = row.get('size', '')
        candidate_size_norm = normalize_size_for_matching(candidate_size)
        if candidate_size_norm == input_size_norm:
            size_candidates.append(idx)
    
    if not size_candidates:
        print("No size matches found after brand filtering")
        return {'most_identical': [], 'closely_related': [], 'total_matches': 0}
    
    final_candidates_df = other_stores_df.loc[size_candidates].copy()
    print(f"After size filtering: {len(final_candidates_df)} products")
    
    # Step 2: Similarity Scoring on Product Title Only
    try:
        # Create signatures for remaining candidates (title only)
        candidate_signatures = []
        valid_candidates = []
        
        for idx, row in final_candidates_df.iterrows():
            signature = create_product_signature(row)
            if signature.strip():
                candidate_signatures.append(signature)
                valid_candidates.append(idx)
        
        if not candidate_signatures:
            return {'most_identical': [], 'closely_related': [], 'total_matches': 0}
        
        # Generate embeddings for input and candidates
        all_signatures = [input_signature] + candidate_signatures
        embeddings = model.encode(all_signatures)
        
        # Calculate similarities
        input_embedding = embeddings[0].reshape(1, -1)
        candidate_embeddings = embeddings[1:]
        similarities = cosine_similarity(input_embedding, candidate_embeddings)[0]
        
        high_threshold = 0.85  
        
        most_identical_matches = []
        closely_related_matches = []
        
        for idx, similarity_score in enumerate(similarities):
            if similarity_score >= high_threshold:
                match_category = "most_identical"
                matches_list = most_identical_matches
            else:
                match_category = "closely_related"
                matches_list = closely_related_matches
            
            candidate_idx = valid_candidates[idx]
            matching_row = other_stores_df.loc[candidate_idx]
            
            match_info = {
                'product_title': matching_row.get('productTitle', 'N/A'),
                'brand_name': matching_row.get('brandName', 'N/A'),
                'size': matching_row.get('size', 'N/A'),
                'price': matching_row.get('productPrice', 'N/A'),
                'store': matching_row.get('shop', 'N/A'),
                'category': matching_row.get('mainCategoryName', 'N/A'),
                'subcategory': matching_row.get('subCategoryName', 'N/A'),
                'similarity_score': round(similarity_score, 4),
                'match_category': match_category,
                'signature': candidate_signatures[idx]
            }
            matches_list.append(match_info)
        
        # Sort matches by similarity score
        most_identical_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        closely_related_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Filter to keep only the highest scoring match per store
        def keep_best_match_per_store(matches):
            """Keep only the match with highest similarity score per store"""
            store_best_matches = {}
            for match in matches:
                store = match['store']
                if store not in store_best_matches or match['similarity_score'] > store_best_matches[store]['similarity_score']:
                    store_best_matches[store] = match
            return list(store_best_matches.values())
        
        # Apply filtering to both categories
        most_identical_matches = keep_best_match_per_store(most_identical_matches)
        closely_related_matches = keep_best_match_per_store(closely_related_matches)
        
        total_matches = len(most_identical_matches) + len(closely_related_matches)
        print(f"Final matches found: {total_matches} (Most identical: {len(most_identical_matches)}, Closely related: {len(closely_related_matches)})")
        # Print similarity scores for most identical matches
        if most_identical_matches:
            print("\n=== Most Identical Products similarity) ===")
            for match in most_identical_matches:
                print(f"Store: {match['store']}, Similarity: {match['similarity_score']:.4f}")

        return {
            'most_identical': most_identical_matches,
            'closely_related': closely_related_matches,
            'total_matches': total_matches
        }
        
    except Exception as e:
        st.error(f"Error in similarity calculation: {e}")
        return {'most_identical': [], 'closely_related': [], 'total_matches': 0}

@st.cache_data
def load_and_process_data():
    """Load and process the data with caching"""
    data_dirs = [
        "Softdrinks_Chocolate_Products (1)/Softdrinks_Chocolate_Products/Coles(Softdrinks_Chocolate)",
        "Softdrinks_Chocolate_Products (1)/Softdrinks_Chocolate_Products/Wools(Softdrinks_Chocolate)",
        "Softdrinks_Chocolate_Products (1)/Softdrinks_Chocolate_Products/Iga(Softdrinks_Chocolate)"
    ]
    
    datasets = {}
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            datasets.update(load_csv_datasets(data_dir))
    
    if not datasets:
        return None
    
    columns_to_keep = ['productTitle', 'productPrice', 'shop', 'brandName', 'mainCategoryName','subCategoryName']

    for name, df in datasets.items():
        # Filter to only keep columns that exist in the dataframe
        available_columns = [col for col in columns_to_keep if col in df.columns]
        datasets[name] = df[available_columns]
    
    combined_df = pd.concat(datasets.values(), ignore_index=True)
    
    # Process the data
    combined_df['productTitle'] = combined_df['productTitle'].apply(normalize_product_title)
    
    # Extract size and clean titles
    size_and_title = combined_df['productTitle'].apply(extract_size_and_clean_title)
    combined_df['size'] = [item[0] for item in size_and_title]
    combined_df['productTitle'] = [item[1] for item in size_and_title]
    
    # Normalize size field
    combined_df['size'] = combined_df['size'].apply(normalize_size_field)
    
    # Normalize brand names
    if 'brandName' in combined_df.columns:
        combined_df['brandName'] = combined_df['brandName'].apply(normalize_brand_name_fuzzy)
    
    # Write the modified dataset to output.csv
    combined_df.to_csv('output.csv', index=False)
    
    return combined_df

def display_matches_in_streamlit(matches, input_product):
    """Display matching products in Streamlit format with enhanced information"""
    
    if matches['total_matches'] > 0:
        st.success(f"🎯 Found {matches['total_matches']} highly accurate matches in other stores!")
        
        # Show input product details for reference
        with st.expander("🔍 Reference Product Details", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Brand:** {input_product.get('brandName', 'N/A')}")
                st.write(f"**Title:** {input_product.get('productTitle', 'N/A')}")
            with col2:
                st.write(f"**Size:** {input_product.get('size', 'N/A')}")
                st.write(f"**Store:** {input_product.get('shop', 'N/A')}")
            with col3:
                st.write(f"**Price:** ${input_product.get('productPrice', 'N/A')}")
        
        # Display Most Identical Products (85%+)
        if matches['most_identical']:
            st.subheader(f"Identical Products - {len(matches['most_identical'])} found")
            st.info("These products passed strict brand and size matching")
            
            for i, match in enumerate(matches['most_identical'], 1):
                with st.expander(f"#{i} {match['store']}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Product Details:**")
                        st.write(f" Product: {match['product_title']}")
                        st.write(f" Brand: {match['brand_name']}")
                        st.write(f" Size: {match['size']}")
                    
                    with col2:
                        st.write("**Store & Price:**")
                        st.write(f" Store: {match['store']}")
                        st.write(f" Price: ${match['price']}")
                        
                        # Calculate price difference
                        try:
                            input_price = float(input_product.get('productPrice', 0))
                            match_price = float(match['price'])
                            price_diff = match_price - input_price
                            if price_diff > 0:
                                st.write(f" +${price_diff:.2f} more expensive")
                            elif price_diff < 0:
                                st.write(f" ${abs(price_diff):.2f} cheaper")
                            else:
                                st.write("💱 Same price")
                        except (ValueError, TypeError):
                            pass
                    
                    with col3:
                        st.write("**Category:**")
                        st.write(f" Category: {match['category']}")
                        st.write(f" Subcategory: {match['subcategory']}")
        
        # Display Closely Related Products (candidates that passed filtering)
        if matches['closely_related']:
            st.subheader(f" Closely Related Products (candidates after filtering) - {len(matches['closely_related'])} found")
            st.info("These products have the same brand and size, but lower title similarity.")
            
            for i, match in enumerate(matches['closely_related'], 1):
                with st.expander(f"#{i} {match['store']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Product Details:**")
                        st.write(f" Product: {match['product_title']}")
                        st.write(f" Brand: {match['brand_name']}")
                        st.write(f" Size: {match['size']}")
                    
                    with col2:
                        st.write("**Store & Price:**")
                        st.write(f"Store: {match['store']}")
                        st.write(f"Price: ${match['price']}")
                        
                        # Calculate price difference
                        try:
                            input_price = float(input_product.get('productPrice', 0))
                            match_price = float(match['price'])
                            price_diff = match_price - input_price
                            if price_diff > 0:
                                st.write(f" +${price_diff:.2f} more expensive")
                            elif price_diff < 0:
                                st.write(f" ${abs(price_diff):.2f} cheaper")
                            else:
                                st.write("💱 Same price")
                        except (ValueError, TypeError):
                            pass
                    
                    with col3:
                        st.write("**Category:**")
                        st.write(f" Category: {match['category']}")
                        st.write(f" Subcategory: {match['subcategory']}")
    else:
        st.warning(" No highly accurate matches found.")
        st.info(" This means no products in other stores have the same brand AND size with sufficient title similarity (85%+).")
        st.markdown("**Possible reasons:**")
        st.markdown("- Product not available in other stores")
        st.markdown("- Different product naming conventions")
        st.markdown("- Different size offerings across stores")

def main():
    st.title("🛒 Accurate Product Matching Across Stores")
    st.markdown("Find **highly accurate** identical products across different grocery stores using advanced hybrid matching!")
    
    # Add info about the improved matching approach
    with st.expander("ℹ️ How Our Improved Matching Works", expanded=False):
        st.markdown("""
        **Our hybrid approach ensures high accuracy by:**
        1. ** Strict Filtering**: Only compares products with matching brand (fuzzy) and exact size
        2. ** Smart Similarity**: Uses AI to compare product titles of pre-filtered candidates
        3. ** High Threshold**: Only shows matches with 85%+ similarity for maximum accuracy
        
        **This prevents false matches like:**
        -  Coca-Cola 1.25L matching with Coca-Cola 2L (different sizes)
        -  Coca-Cola matching with Pepsi (different brands)
        -  Only matches truly identical products across stores
        """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📊 Data Statistics")
        
        # Load data button
        if st.button("🔄 Reload Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    with st.spinner("Loading and processing data..."):
        combined_df = load_and_process_data()
    
    if combined_df is None:
        st.error(" Could not load data. Please check if the data directories exist.")
        st.info("Expected directories:")
        st.code("""
        Softdrinks_Chocolate_Products (1)/Softdrinks_Chocolate_Products/Coles(Softdrinks_Chocolate)
        Softdrinks_Chocolate_Products (1)/Softdrinks_Chocolate_Products/Wools(Softdrinks_Chocolate)
        Softdrinks_Chocolate_Products (1)/Softdrinks_Chocolate_Products/Iga(Softdrinks_Chocolate)
        """)
        return
    
    # Update sidebar with statistics
    with st.sidebar:
        st.metric("Total Products", len(combined_df))
        if 'shop' in combined_df.columns:
            stores = combined_df['shop'].unique()
            st.metric("Number of Stores", len(stores))
            st.write("**Stores:**")
            for store in stores:
                count = len(combined_df[combined_df['shop'] == store])
                st.write(f"• {store}: {count} products")
    
    st.success(f" Loaded {len(combined_df)} products successfully!")
    
    # Display the dataframe
    st.header(" Product Database")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'shop' in combined_df.columns:
            selected_stores = st.multiselect(
                "Filter by Store:",
                options=combined_df['shop'].unique(),
                default=combined_df['shop'].unique()
            )
        else:
            selected_stores = []
    
    with col2:
        if 'brandName' in combined_df.columns:
            brands = combined_df['brandName'].dropna().unique()
            selected_brands = st.multiselect(
                "Filter by Brand:",
                options=brands,
                default=[]
            )
        else:
            selected_brands = []
    
    with col3:
        search_term = st.text_input("🔍 Search Products:", placeholder="Enter product name...")
    
    # Apply filters
    filtered_df = combined_df.copy()
    
    if selected_stores:
        filtered_df = filtered_df[filtered_df['shop'].isin(selected_stores)]
    
    if selected_brands:
        filtered_df = filtered_df[filtered_df['brandName'].isin(selected_brands)]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['productTitle'].str.contains(search_term, case=False, na=False)
        ]
    
    # Display filtered dataframe with selection
    st.subheader(f"Showing {len(filtered_df)} products")
    
    if len(filtered_df) > 0:
        # Create a selection dataframe for display
        display_df = filtered_df.copy()
        display_df.index = range(len(display_df))
        
        # Display the dataframe
        selected_indices = st.dataframe(
            display_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Check if a row is selected
        if selected_indices.selection.rows:
            selected_row_idx = selected_indices.selection.rows[0]
            selected_product = display_df.iloc[selected_row_idx]
            
            st.markdown("---")
            st.header(" Selected Product Details")
            
            # Display selected product info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Product Information:**")
                st.write(f" **Brand:** {selected_product.get('brandName', 'N/A')}")
                st.write(f" **Product:** {selected_product.get('productTitle', 'N/A')}")
                st.write(f" **Size:** {selected_product.get('size', 'N/A')}")
            
            with col2:
                st.write("**Store & Price:**")
                st.write(f" **Store:** {selected_product.get('shop', 'N/A')}")
                st.write(f" **Price:** ${selected_product.get('productPrice', 'N/A')}")
            
            with col3:
                st.write("**Category:**")
                st.write(f" **Category:** {selected_product.get('mainCategoryName', 'N/A')}")
                st.write(f" **Subcategory:** {selected_product.get('subCategoryName', 'N/A')}")
            # Find matching products button
            if st.button(" Find Accurate Matches", type="primary"):
                with st.spinner("🔍 Finding highly accurate matches using hybrid approach..."):
                    matches = find_identical_products_across_stores(
                        selected_product, 
                        combined_df
                    )
                
                st.markdown("---")
                st.header("📊 Accurate Matching Results")
                display_matches_in_streamlit(matches, selected_product)
        else:
            st.info(" Please select a row from the table above to find matching products.")
    else:
        st.warning("No products match your current filters.")

if __name__ == "__main__":
    main()
