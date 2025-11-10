"""
scripts/data_merge.py
Author: Mariam Awini Issah
Purpose: Load raw social and transaction datasets, clean, handle duplicates, merge, 
         and save processed dataset for Task 1.
"""

import pandas as pd
import os

# --- Step 1: File paths ---
RAW_SOCIAL = "C:/Users/awini/formative2-mlp/data/raw/customer_social_profiles.csv"
RAW_TRANS = "C:/Users/awini/formative2-mlp/data/raw/customer_transactions.csv"
PROCESSED_OUT = "C:/Users/awini/formative2-mlp/data/processed/merged_customer_data.csv"

# --- Step 2: Load raw data ---
social = pd.read_csv(RAW_SOCIAL)
trans = pd.read_csv(RAW_TRANS)

# Strip whitespace from column names
social.columns = social.columns.str.strip()
trans.columns = trans.columns.str.strip()

# --- Step 3: Clean IDs ---
# Social IDs: extract numeric part (A178 -> 178)
social['customer_id_new'] = social['customer_id_new'].astype(str).str.extract(r'(\d+)').astype(int)
# Transaction IDs: ensure numeric type
trans['customer_id_new'] = trans['customer_id_new'].astype(int)

# --- Step 4: Handle duplicates ---
# Aggregate social data to one row per customer
social_agg = social.groupby('customer_id_new').agg({
    'social_media_platform': lambda x: ','.join(x.unique()),  # concatenate multiple platforms
    'engagement_score': 'mean',
    'purchase_interest_score': 'mean',
    'review_sentiment': lambda x: ','.join(x.unique())
}).reset_index()

# --- Step 5: Merge datasets ---
merged_df = pd.merge(
    social_agg,
    trans,
    on='customer_id_new',
    how='inner'  # keep only customers with both social and transaction info
)

# --- Step 6: Save processed dataset ---
os.makedirs(os.path.dirname(PROCESSED_OUT), exist_ok=True)
merged_df.to_csv(PROCESSED_OUT, index=False)

print(f"✅ Merged dataset saved at '{PROCESSED_OUT}'")