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
        r'\bx\s*(\d+)': r'pack of \1',
        r'(\d+)\s*x\s*(\d+)': r'\1 pack of \2',
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
        
        # Size x count: "375ml x 6" or "375ml x6"
        (r'\b(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)\s*[xX×]\s*(\d+)',
         lambda m: f"{m.group(1)}{m.group(2)}x{m.group(3)}"),
        
        # Count x size: "6 x 375ml"
        (r'\b(\d+)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(millilitre|ml|litre|l|gram|g|kilogram|kg)',
         lambda m: f"{m.group(2)}{m.group(3)}x{m.group(1)}"),
        
        # Simple size: "375ml", "1.25l", "45g"
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
    Creating a text signature for product matching using key attributes.
    This combines brand, product title, and size information.
    """
    signature_parts = []
    
    # Add brand name 
    if pd.notna(row.get('brandName', '')) and row.get('brandName', '').strip():
        signature_parts.append(str(row['brandName']).strip())
    
    # Add cleaned product title
    if pd.notna(row.get('productTitle', '')) and row.get('productTitle', '').strip():
        signature_parts.append(str(row['productTitle']).strip())
    
    # Add size information (important for exact matching)
    if pd.notna(row.get('size', '')) and row.get('size', '').strip():
        signature_parts.append(str(row['size']).strip())
    
    # Join with spaces to create signature
    signature = ' '.join(signature_parts)
    return signature.lower().strip()

@st.cache_resource
def load_sentence_transformer():
    """Cache the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def find_identical_products_across_stores(input_row, dataframe):
    """
    Find identical products across different stores using vector embeddings and cosine similarity.
    """
    model = load_sentence_transformer()
    
    # Create signature for input product
    input_signature = create_product_signature(input_row)
    input_store = input_row.get('shop', 'Unknown')
    
    if not input_signature:
        return {'near_identical': [], 'closely_related': [], 'total_matches': 0}
    
    # Filter out products from the same store
    other_stores_df = dataframe[dataframe['shop'] != input_store].copy()
    
    if len(other_stores_df) == 0:
        return {'near_identical': [], 'closely_related': [], 'total_matches': 0}
    
    # Create signatures for all products from other stores
    other_signatures = other_stores_df.apply(create_product_signature, axis=1).tolist()
    
    # Remove empty signatures
    valid_indices = [i for i, sig in enumerate(other_signatures) if sig.strip()]
    if not valid_indices:
        return {'near_identical': [], 'closely_related': [], 'total_matches': 0}
    
    valid_other_df = other_stores_df.iloc[valid_indices].copy()
    valid_signatures = [other_signatures[i] for i in valid_indices]
    
    # Generate embeddings
    try:
        # Combine input signature with all other signatures for batch processing
        all_signatures = [input_signature] + valid_signatures
        embeddings = model.encode(all_signatures)
        
        # Separate input embedding from other embeddings
        input_embedding = embeddings[0].reshape(1, -1)
        other_embeddings = embeddings[1:]
        
        # Calculate cosine similarities
        similarities = cosine_similarity(input_embedding, other_embeddings)[0]
        
        # Categorize matches by similarity levels
        near_identical_matches = []  # 95%+ similarity
        closely_related_matches = []  # 80-95% similarity
        
        for idx, similarity_score in enumerate(similarities):
            if similarity_score >= 0.95:  # 95%+ = Near-identical
                match_category = "near_identical"
                matches_list = near_identical_matches
            elif similarity_score >= 0.80:  # 80-95% = Closely related
                match_category = "closely_related" 
                matches_list = closely_related_matches
            else:
                continue  # Skip matches below 80%
            
            matching_row = valid_other_df.iloc[idx]
            
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
                'signature': valid_signatures[idx]
            }
            matches_list.append(match_info)
        
        # Sort each category by similarity score (highest first)
        near_identical_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        closely_related_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return categorized results
        return {
            'near_identical': near_identical_matches,
            'closely_related': closely_related_matches,
            'total_matches': len(near_identical_matches) + len(closely_related_matches)
        }
        
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return {'near_identical': [], 'closely_related': [], 'total_matches': 0}

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
    
    return combined_df

def display_matches_in_streamlit(matches, input_product):
    """Display matching products in Streamlit format"""
    
    if matches['total_matches'] > 0:
        st.success(f" Found {matches['total_matches']} matching products in other stores!")
        
        # Display Near-Identical Products (95%+)
        if matches['near_identical']:
            st.subheader(f" Near-Identical Products (95%+ similarity) - {len(matches['near_identical'])} found")
            
            for i, match in enumerate(matches['near_identical'], 1):
                with st.expander(f"#{i} {match['store']} - Similarity: {match['similarity_score']:.1%}", expanded=True):
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
                    
                    with col3:
                        st.write("**Category:**")
                        st.write(f" Category: {match['category']}")
                        st.write(f" Subcategory: {match['subcategory']}")
        
        # Display Closely Related Products (80-95%)
        if matches['closely_related']:
            st.subheader(f"🔍 Closely Related Products (80-95% similarity) - {len(matches['closely_related'])} found")
            
            for i, match in enumerate(matches['closely_related'], 1):
                with st.expander(f"#{i} {match['store']} - Similarity: {match['similarity_score']:.1%}"):
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
                    
                    with col3:
                        st.write("**Category:**")
                        st.write(f" Category: {match['category']}")
                        st.write(f" Subcategory: {match['subcategory']}")
    else:
        st.warning(" No matching products found in other stores with the current threshold.")

def main():
    st.title("🛒 Product Matching Across Stores")
    st.markdown("Find identical and similar products across different grocery stores!")
    
    # Sidebar for controls
    with st.sidebar:
        st.header(" Data Statistics")
        
        # Load data button
        if st.button(" Reload Data"):
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
                        # Find matchingproducts button
            if st.button(" Find Matching Products", type="primary"):
                with st.spinner(" Finding matching products across stores..."):
                    matches = find_identical_products_across_stores(
                        selected_product, 
                        combined_df
                    )
                
                st.markdown("---")
                st.header(" Matching Products Results")
                display_matches_in_streamlit(matches, selected_product)
        else:
            st.info(" Please select a row from the table above to find matching products.")
    else:
        st.warning("No products match your current filters.")

if __name__ == "__main__":
    main()
