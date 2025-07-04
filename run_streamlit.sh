#!/bin/bash

# Activate the virtual environment and run the Streamlit app
source grocery_app_env/bin/activate
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
