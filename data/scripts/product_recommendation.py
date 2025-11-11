import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Paths
MODELS_DIR = "C:/Users/awini/formative2-mlp/data/processed/models"
BEST_MODEL_INFO_FP = os.path.join(MODELS_DIR, 'best_model_info.json')


# Load best model info
with open(BEST_MODEL_INFO_FP, 'r') as f:
    best_model_info = json.load(f)

# Load label encoder
le = joblib.load(best_model_info['label_encoder'])


# Load models
def load_models():
    models = {}
    
    # Random Forest
    rf_fp = os.path.join(MODELS_DIR, "product_recommender_rf.joblib")
    if os.path.exists(rf_fp):
        models['RandomForest'] = joblib.load(rf_fp)
    
    # Logistic Regression
    lr_fp = os.path.join(MODELS_DIR, "product_recommender_lr.joblib")
    if os.path.exists(lr_fp):
        models['LogisticRegression'] = joblib.load(lr_fp)
    
    # XGBoost
    xgb_fp = os.path.join(MODELS_DIR, "product_recommender_xgb.joblib")
    if os.path.exists(xgb_fp):
        models['XGBoost'] = joblib.load(xgb_fp)
    
    # Keras
    keras_fp = os.path.join(MODELS_DIR, "product_recommender_keras.keras")
    if os.path.exists(keras_fp):
        models['KerasNN'] = load_model(keras_fp)
    
    return models

MODELS = load_models()


# Define preprocessing
def build_preprocessor(numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor

# Prediction function
def recommend_product(customer_df, numeric_cols, categorical_cols):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(customer_df)  # fit on current data
    
    X_input = preprocessor.transform(customer_df)
    
    results = {}
    
    for name, model in MODELS.items():
        if name == 'KerasNN':
            yprob = model.predict(X_input)
            y_pred = np.argmax(yprob, axis=1)
        else:
            y_pred = model.predict(X_input)
        # Convert back to category
        results[name] = le.inverse_transform(y_pred)
    
    # Optionally return best model prediction
    best_name = best_model_info['best_model_name']
    results['BestModel'] = results.get(best_name)
    
    return results

# Example usage
if __name__ == "__main__":
    # Load sample data for testing
    DATA_FP = "C:/Users/awini/formative2-mlp/data/processed/merged_customer_data.csv"
    df = pd.read_csv(DATA_FP)
    
    # Columns used in preprocessing
    numeric_cols = ['engagement_score', 'purchase_interest_score', 'purchase_amount', 'customer_rating']
    categorical_cols = ['primary_platform', 'review_sentiment']
    
    sample_customer = df.sample(1)
    predictions = recommend_product(sample_customer, numeric_cols, categorical_cols)
    print("Predictions for sample customer:")
    print(predictions)