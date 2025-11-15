"""Model loading utilities for the product recommendation system."""

import json
from pathlib import Path
from typing import Any, Dict
import joblib  # type: ignore
import numpy as np  # type: ignore
import librosa  # type: ignore
import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image  # type: ignore

from tensorflow.keras.models import load_model  # type: ignore

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS = Path("models")


# ==================== MODEL LOADERS ====================
@st.cache_resource
def load_product_model() -> Dict[str, Any]:
    """
    Load the XGBoost product recommendation model.
    
    Returns:
        Dictionary containing product model and metadata
    """
    result = {
        'model': None,
        'model_used': None,
        'label_encoder': None,
        'model_info': None,
    }

    xgb_path = MODELS / "product_recommender_xgb.joblib"
    label_encoder_path = MODELS / "label_encoder.joblib"

    if xgb_path.exists():
        try:
            result['model'] = joblib.load(xgb_path) # type: ignore
            result['model_used'] = 'XGBoost' # type: ignore
        except FileNotFoundError:
            # st.error(f"Failed to load XGBoost model: {e}")
            result['model'] = None
            result['model_used'] = 'Failed' # type: ignore
    else:
        # st.warning(f"XGBoost model not found at {xgb_path}")
        result['model_used'] = 'Not Found' # type: ignore
    
    # Load label encoder
    if label_encoder_path.exists():
        try:
            result['label_encoder'] = joblib.load(label_encoder_path) # type: ignore
        except FileNotFoundError:
            result['label_encoder'] = None
    else:
        result['label_encoder'] = None

    return result

@st.cache_resource
def load_face_model(): # type: ignore
    """Load face model ONLY when called (lazy loading)"""
    face_model_path = MODELS / "face_classification_model.keras"
    class_names_path = MODELS / "face_classification_model_class_names.npy"

    if not face_model_path.exists() or not class_names_path.exists():
        return None, None

    try:
        # Show loading message
        with st.spinner("Loading face recognition model (first time only)..."):
            face_model = load_model(face_model_path, compile=False)  # compile=False is faster # type: ignore
            class_names = np.load(class_names_path)

        return face_model, class_names # type: ignore
    except Exception as e:
        st.error(f"Failed to load face model: {e}")
        return None, None

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
    voice_model_path = MODELS / "audio_model.pkl"
    feature_names_path = MODELS / "feature_names.pkl"
    class_names_path = MODELS / "class_names.pkl"

    if not all([voice_model_path.exists(),
                feature_names_path.exists(),
                class_names_path.exists()]):
        return None, None, None


    try:
        voice_model = joblib.load(voice_model_path)  # type: ignore
        feature_names = joblib.load(feature_names_path)  # type: ignore
        class_names = joblib.load(class_names_path)  # type: ignore

        return voice_model, feature_names, class_names
    except Exception:
        st.error("Error loading voice model files.")
        return None, None, None


# ==================== AUDIO PROCESSING ====================

def extract_audio_features(audio_file) -> dict | None:  # type: ignore
    """
    Extract comprehensive audio features from audio file for voice verification.
    
    Extracts 97+ features including:
    - MFCCs (52 features: 13 coefficients × 4 stats)
    - Spectral features (16 features)
    - Energy features (8 features)
    - Zero Crossing Rate (2 features)
    - Spectral Centroid (2 features)
    - Chroma features (12 features)
    - Pitch/F0 features (4 features)
    - Delta MFCCs (13 features)
    
    Args:
        audio_file: Uploaded audio file
    
    Returns:
        Dictionary of features or None if extraction failed
    """
    try:
        # Reset file pointer for Streamlit uploaded files
        if hasattr(audio_file, 'seek'):
            audio_file.seek(0)

        # Load audio file (matching training sample rate)
        y, sr = librosa.load(audio_file, sr=44100)  # type: ignore

        # Initialize features dictionary
        features = {}

        # ==================== 1. MFCCs (52 features) ====================
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # type: ignore

        for i in range(13):
            mfcc_num = i + 1
            features[f'mfcc{mfcc_num}_mean'] = np.mean(mfccs[i])  # type: ignore
            features[f'mfcc{mfcc_num}_std'] = np.std(mfccs[i])  # type: ignore
            features[f'mfcc{mfcc_num}_min'] = np.min(mfccs[i])  # type: ignore
            features[f'mfcc{mfcc_num}_max'] = np.max(mfccs[i])  # type: ignore

        # ==================== 2. Spectral Rolloff (4 features) ====================
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # type: ignore
        features['rolloff_mean'] = np.mean(rolloff)  # type: ignore
        features['rolloff_std'] = np.std(rolloff)  # type: ignore
        features['rolloff_min'] = np.min(rolloff)  # type: ignore
        features['rolloff_max'] = np.max(rolloff)  # type: ignore

        # ==================== 3. RMS Energy (4 features) ====================
        rms_energy = librosa.feature.rms(y=y)  # type: ignore
        features['energy_mean'] = np.mean(rms_energy)  # type: ignore
        features['energy_std'] = np.std(rms_energy)  # type: ignore
        features['energy_min'] = np.min(rms_energy)  # type: ignore
        features['energy_max'] = np.max(rms_energy)  # type: ignore

        # ==================== 4. Zero Crossing Rate (2 features) ====================
        zcr = librosa.feature.zero_crossing_rate(y)  # type: ignore
        features['zcr_mean'] = np.mean(zcr)  # type: ignore
        features['zcr_std'] = np.std(zcr)  # type: ignore

        # ==================== 5. Spectral Centroid (2 features) ====================
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # type: ignore
        features['centroid_mean'] = np.mean(centroid)  # type: ignore
        features['centroid_std'] = np.std(centroid)  # type: ignore

        # ==================== TOTAL SO FAR: 68 features ====================
        # This matches your training data, so the code above stays the same

        # ==================== 6. NEW: Chroma Features (12 features) ====================
        # Represents pitch class distribution (useful for voice timbre)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # type: ignore
            chroma_mean = np.mean(chroma, axis=1)  # type: ignore

            for i in range(12):
                features[f'chroma{i+1}_mean'] = chroma_mean[i]  # type: ignore
        except Exception:
            # If chroma extraction fails, use zeros
            for i in range(12):
                features[f'chroma{i+1}_mean'] = 0.0

        # ==================== 7. NEW: Pitch/F0 Features (4 features) ====================
        # Fundamental frequency - very distinctive for different speakers
        try:
            # Use piptrack for F0 estimation
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)  # type: ignore

            # Get pitch values where magnitude is highest
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()  # type: ignore
                pitch = pitches[index, t]
                if pitch > 0:  # Valid pitch
                    pitch_values.append(pitch)

            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)  # type: ignore
                features['pitch_std'] = np.std(pitch_values)  # type: ignore
                features['pitch_min'] = np.min(pitch_values)  # type: ignore
                features['pitch_max'] = np.max(pitch_values)  # type: ignore
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_min'] = 0.0
                features['pitch_max'] = 0.0
        except Exception:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0

        # ==================== 8. NEW: Delta MFCCs (13 features) ====================
        # Captures how MFCCs change over time (temporal dynamics)
        try:
            delta_mfccs = librosa.feature.delta(mfccs)  # type: ignore
            delta_mean = np.mean(delta_mfccs, axis=1)  # type: ignore

            for i in range(13):
                features[f'delta_mfcc{i+1}_mean'] = delta_mean[i]  # type: ignore
        except Exception:
            for i in range(13):
                features[f'delta_mfcc{i+1}_mean'] = 0.0

        # ==================== 9. NEW: Spectral Bandwidth (4 features) ====================
        # Measures the width of the frequency spectrum
        try:
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # type: ignore
            features['bandwidth_mean'] = np.mean(bandwidth)  # type: ignore
            features['bandwidth_std'] = np.std(bandwidth)  # type: ignore
            features['bandwidth_min'] = np.min(bandwidth)  # type: ignore
            features['bandwidth_max'] = np.max(bandwidth)  # type: ignore
        except Exception:
            features['bandwidth_mean'] = 0.0
            features['bandwidth_std'] = 0.0
            features['bandwidth_min'] = 0.0
            features['bandwidth_max'] = 0.0

        # ==================== 10. NEW: Spectral Contrast (4 features) ====================
        # Difference between peaks and valleys in the spectrum
        try:
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # type: ignore
            features['contrast_mean'] = np.mean(contrast)  # type: ignore
            features['contrast_std'] = np.std(contrast)  # type: ignore
            features['contrast_min'] = np.min(contrast)  # type: ignore
            features['contrast_max'] = np.max(contrast)  # type: ignore
        except Exception:
            features['contrast_mean'] = 0.0
            features['contrast_std'] = 0.0
            features['contrast_min'] = 0.0
            features['contrast_max'] = 0.0

        # ==================== TOTAL: 68 + 12 + 4 + 13 + 4 + 4 = 105 features ====================

        return features  # type: ignore
    
    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Audio feature extraction error: {e}")
        print(traceback.format_exc())
        return None


# ==================== Voice prediction ====================
def predict_speaker(voice_model, feature_names, class_names, audio_features): # type: ignore
    """
    Predict the speaker from audio features.
    
    Args:
        voice_model: Trained Random Forest model
        feature_names: List of feature column names (68 features)
        class_names: List of speaker names
        audio_features: Dictionary of extracted features
    
    Returns:
        predicted_speaker: Name of the predicted speaker
        confidence: Probability/confidence of prediction
    """
    try:
        # Convert features dict to DataFrame with correct column order
        features_df = pd.DataFrame([audio_features])

        # Reorder columns to match training data
        features_df = features_df[feature_names]  # type: ignore

        # Make prediction
        prediction = voice_model.predict(features_df)[0] # type: ignore

        # Get prediction probabilities for all classes
        prediction_proba = voice_model.predict_proba(features_df)[0] # type: ignore

        # Find the index of the predicted class
        predicted_idx = np.argmax(prediction_proba) # type: ignore
        confidence = prediction_proba[predicted_idx] # type: ignore

        # Get speaker name
        predicted_speaker = class_names[predicted_idx] # type: ignore

        return predicted_speaker, confidence, prediction_proba # type: ignore

    except Exception:
        return None, None, None


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
    face_model, face_class_names = load_face_model()  # type: ignore
    voice_model, feature_names, voice_class_names = load_voice_model()

    # Combine into single dictionary
    all_models = { # type: ignore
        # Product recommendation
        'product_model': product_data['model'],
        'model_used': product_data['model_used'],
        'label_encoder': product_data['label_encoder'],
        'model_info': product_data['model_info'],

        # Face Authentication models
        'face_model': face_model,
        'face_class_names': face_class_names,

        # Voice verification model
        'voice_model': voice_model,
        'feature_names': feature_names,
        'voice_class_names': voice_class_names,
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
