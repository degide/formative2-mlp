"""Model loading utilities for the product recommendation system."""

import json
import os
from pathlib import Path
from typing import Any, Dict
import joblib  # type: ignore
import numpy as np  # type: ignore
import librosa  # type: ignore
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
# import tensorflow as tf  # type: ignore
# from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image  # type: ignore

# Configure TensorFlow environment BEFORE importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''    # Force CPU-only mode

# Paths
DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "processed" / "models"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS = Path("models")

# ==================== HELPER FUNCTIONS ====================

def load_model_file(file_path: Path, model_name: str):
    """
    Load a single model file with error handling.
    
    Args:
        file_path: Path to the model file
        model_name: Name of the model for error messages
    
    Returns:
        Loaded model or None if failed
    """
    try:
        model = joblib.load(file_path)  # type: ignore
        # st.success(f"{model_name} loaded successfully")
        return model
    except FileNotFoundError:
        # st.warning(f"{model_name} not found at {file_path}")
        return None
    except Exception:
        # st.error(f"Error loading {model_name}: {str(e)[:100]}")
        return None


def load_best_model_info() -> Dict[str, Any]:
    """
    Load the best model information from JSON file.
    
    Returns:
        Dictionary containing best model info
    """
    try:
        with open(MODEL_DIR / "best_model_info.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # st.warning("best_model_info.json not found, using Random Forest as default")
        return {'best_model_name': 'Random Forest'}
    except json.JSONDecodeError:
        # st.error(f"Error reading best_model_info.json: {e}")
        return {'best_model_name': 'Random Forest'}


# ==================== MODEL LOADERS ====================

@st.cache_resource
def load_product_model() -> Dict[str, Any]:
    """
    Load the product recommendation model with fallback logic.
    
    Returns:
        Dictionary containing product model and metadata
    """
    result = {
        'model': None,
        'model_used': None,
        'label_encoder': None,
        'model_info': None
    }

    # Load model info
    model_info = load_best_model_info()
    result['model_info'] = model_info # type: ignore
    best_model_name = model_info.get('best_model_name', 'Random Forest')

    # Try to load the best model first
    if best_model_name == 'XGBoost':
        xgb_model = load_model_file(
            MODEL_DIR / "product_recommender_xgb.joblib",
            "XGBoost Product Model"
        )

        if xgb_model is not None:
            result['model'] = xgb_model
            result['model_used'] = 'XGBoost' # type: ignore
        else:
            # Fallback to Random Forest
            # st.info("Falling back to Random Forest model...")
            rf_model = load_model_file(
                MODEL_DIR / "product_recommender_rf.joblib",
                "Random Forest Product Model"
            )
            result['model'] = rf_model
            result['model_used'] = 'Random Forest (fallback)' # type: ignore
    else:
        # Load Random Forest directly
        rf_model = load_model_file(
            MODEL_DIR / "product_recommender_rf.joblib",
            "Random Forest Product Model"
        )
        result['model'] = rf_model
        result['model_used'] = 'Random Forest' # type: ignore

    # Load label encoder
    label_encoder = load_model_file(
        MODEL_DIR / "label_encoder.joblib",
        "Label Encoder"
    )
    result['label_encoder'] = label_encoder

    return result


# @st.cache_resource
# def load_face_model():
#     """Load face model with error handling"""
#     face_model_path = MODELS / "face_classification_model.keras"
#     class_names_path = MODELS / "face_classification_model_class_names.npy"

#     if not face_model_path.exists() or not class_names_path.exists():
#         return None, None

#     try:
#         face_model = load_model(face_model_path)
#         class_names = np.load(class_names_path)

#         return face_model, class_names
#     except Exception as e:
#         return None, None

# PROCESSING FUNCTION FOR FACE IMAGES

def preprocess_face_image(uploaded_file): # type: ignore
    """
    Preprocess uploaded image for face recognition model.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Preprocessed image array ready for model prediction (1, 224, 224, 3)
    """
    try:
        # Open image
        image = Image.open(uploaded_file) # type: ignore

        # Convert to RGB (in case it's RGBA or grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to 224x224 (model input size)
        image = image.resize((224, 224))

        # Convert to numpy array
        img_array = np.array(image)

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # The model has built-in Rescaling layer, so we don't divide by 255
        # img_array = img_array / 255.0

        return img_array

    except Exception:
        return None



@st.cache_resource
def load_voice_model():
    """
    Load the voice verification model.
    
    Returns:
        Voice verification model or None if not available
    """
    # Try different possible extensions
    possible_paths = [
        MODEL_DIR / "voice_verification_model.joblib",
        MODEL_DIR / "voice_verification_model.h5",
        MODEL_DIR / "voice_verification_model.keras",
    ]

    for voice_model_path in possible_paths:
        if voice_model_path.exists():
            return load_model_file(voice_model_path, "Voice Verification Model")

    # st.info("Voice verification model not found - using simulation mode")
    return None


# ==================== AUDIO PROCESSING ====================

def extract_audio_features(audio_file) -> np.ndarray | None: # type: ignore
    """
    Extract MFCC features from audio file for voice verification.
    
    Args:
        audio_file: Uploaded audio file
    
    Returns:
        Numpy array of features or None if extraction failed
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=16000)  # type: ignore

        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # type: ignore

        # Calculate statistics
        mfcc_mean = np.mean(mfccs, axis=1)  # type: ignore
        mfcc_std = np.std(mfccs, axis=1)  # type: ignore

        # Combine features
        features = np.concatenate([mfcc_mean, mfcc_std])

        return features.reshape(1, -1)  # Reshape for model input

    except Exception:
        # st.error(f"Error extracting audio features: {e}")
        return None


# ==================== MAIN MODEL LOADER ====================

@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """
    Load all models (product, face, voice) for the application.
    
    Returns:
        Dictionary containing all loaded models and metadata
    """
    # Load each model separately
    product_data = load_product_model()
    # face_model, class_names = load_face_model() # type: ignore
    voice_model = load_voice_model()

    # Combine into single dictionary
    all_models = { # type: ignore
        # Product recommendation
        'product_model': product_data['model'],
        'model_used': product_data['model_used'],
        'label_encoder': product_data['label_encoder'],
        'model_info': product_data['model_info'],

        # Face Authentication models
        # 'face_model': face_model,
        # 'face_class_names': class_names,

        # Voice verification model
        'voice_model': voice_model,
    }

    return all_models # type: ignore

@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load customer data"""
    try:
        data = pd.read_csv(PROCESSED_DIR / "merged_customer_data.csv")  # type: ignore
        return data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None
