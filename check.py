#!/usr/bin/env python3

# In your terminal or in a Python script
import pandas as pd
import joblib

df = pd.read_csv('data/processed/merged_customer_data.csv')
label_encoder = joblib.load('data/processed/models/label_encoder.joblib')

dataset_categories = set(df['product_category'].unique())
encoder_categories = set(label_encoder.classes_)

print("Categories in dataset:", dataset_categories)
print("Categories in encoder:", encoder_categories)
print("Missing in encoder:", dataset_categories - encoder_categories)
