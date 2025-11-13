"""
Formative 2 - Multimodal Authentication & Product Recommendation System
A Streamlit app implementing:
1. Face Recognition Authentication
2. Voice Verification Authentication  
3. Product Recommendation Model
"""
import json
from pathlib import Path
from typing import Dict, Any
import streamlit as st
import pandas as pd
import joblib  # type: ignore

# Page configuration
st.set_page_config(
    page_title="Recomm",
    page_icon="🛍️",
    layout="wide"
)

# Paths
DATA_DIR = Path("data")
MODEL_DIR = DATA_DIR / "processed" / "models"
PROCESSED_DIR = DATA_DIR / "processed"

# Load models and data
@st.cache_resource
def load_models() -> Dict[str, Any] | None:
    """Load all trained models"""
    loaded_models = {}
    try:
        # Load model info to determine best model
        with open(MODEL_DIR / "best_model_info.json", 'r', encoding='utf-8') as f:
            loaded_models['model_info'] = json.load(f)

        # Try to load XGBoost (best model) first
        best_model_name = loaded_models['model_info'].get('best_model_name', 'Random Forest')

        if best_model_name == 'XGBoost':
            try:
                loaded_models['product_model'] = joblib.load(MODEL_DIR / "product_recommender_xgb.joblib") # type: ignore
                loaded_models['model_used'] = 'XGBoost'
            except FileNotFoundError as e:
                st.warning(f"XGBoost model file not found: {e}")
                loaded_models['product_model'] = joblib.load(MODEL_DIR / "product_recommender_rf.joblib") # type: ignore
                loaded_models['model_used'] = 'Random Forest (fallback)'
            except Exception as e:
                st.warning(f"Error loading XGBoost model: {e}")
                loaded_models['product_model'] = joblib.load(MODEL_DIR / "product_recommender_rf.joblib") # type: ignore
                loaded_models['model_used'] = 'Random Forest (fallback)'
        else:
            loaded_models['product_model'] = joblib.load(MODEL_DIR / "product_recommender_rf.joblib") # type: ignore
            loaded_models['model_used'] = 'Random Forest'

        loaded_models['label_encoder'] = joblib.load(MODEL_DIR / "label_encoder.joblib") # type: ignore

        return loaded_models # type: ignore
    except FileNotFoundError as e:
        st.error(f"Model info file not found: {e}")
    return None

@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load customer data"""
    try:
        data = pd.read_csv(PROCESSED_DIR / "merged_customer_data.csv") # type: ignore
        return data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None

# Initialize session state
if 'face_authenticated' not in st.session_state:
    st.session_state.face_authenticated = False
if 'voice_authenticated' not in st.session_state:
    st.session_state.voice_authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None

# Main title
st.title('🛍️ Product Recommendation System')
st.info('🔐 A sequential authentication system using' \
' facial recognition and voice validation before providing personalized product recommendations')

# Load data and models
df = load_data()
models = load_models()

if df is None or models is None:
    st.stop()

# Sidebar - System Status
with st.sidebar:
    st.header("🔒 Authentication Status")

    # Face authentication status
    if st.session_state.face_authenticated:
        st.success("Face Authenticated")
    else:
        st.error("Face Not Authenticated")

    # Voice authentication status
    if st.session_state.voice_authenticated:
        st.success("Voice Verified")
    else:
        st.error("Voice Not Verified")

    st.divider()

    # Model info
    st.subheader("Model Information")
    st.info(f"**Active Model:** {models['model_used']}")

    with st.expander("🔍 View Model Details"):
        model_features = models['product_model'].feature_names_in_
        st.write("**Model Features:**")
        st.write(model_features)

        st.write("**Label Encoder Classes (Product Categories):**")
        st.write(models['label_encoder'].classes_)

    st.divider()

    # Reset button
    if st.button("Reset Authentication"):
        st.session_state.face_authenticated = False
        st.session_state.voice_authenticated = False
        st.session_state.current_user = None
        st.session_state.prediction = None
        st.session_state.prediction_proba = None
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Overview", "Face Authentication",
      "Product Recommendation", "Voice Confirmation"])

# Tab 1: Data Overview
with tab1:
    st.header("Customer Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df['customer_id_new'].unique())) # type: ignore
    with col2:
        st.metric("Total Transactions", len(df))
    with col3:
        st.metric("Product Categories", len(df['product_category'].unique())) # type: ignore
    with col4:
        st.metric("Avg Purchase Amount", f"${df['purchase_amount'].mean():.2f}")


    with st.expander("View Full Dataset"):
        st.dataframe(df, use_container_width=True)  # type: ignore

    with st.expander("View Data Sample"):
        st.dataframe(df.describe(), use_container_width=True)  # type: ignore

    product_counts = df['product_category'].value_counts()
    with st.expander("View Category Counts"):
        st.bar_chart(product_counts) # type: ignore

# Tab 2: Face Authentication
with tab2:
    st.header("Step 1: Face Recognition Authentication")

    st.write("Upload a facial image for authentication")

    uploaded_image = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png'],
        key="face_upload"
    )

    if uploaded_image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.write("### Authentication Options")

            # Simulated authentication (replace with actual face recognition model)
            auth_choice = st.radio(
                "Simulate authentication result:",
                ["Authorized User", "Unauthorized User"]
            )

            if st.button("Authenticate Face", type="primary"):
                if auth_choice == "Authorized User":
                    st.session_state.face_authenticated = True
                    st.session_state.current_user = "User_001"  # Simulated user ID
                    st.success("Face recognized! Proceed to Product Recommendation tab.")
                else:
                    st.session_state.face_authenticated = False
                    st.error("Face not recognized. Access denied.")
                st.rerun()

    if not st.session_state.face_authenticated:
        st.warning("Please authenticate your face to access product recommendations")
# Tab 3: Product Recommendation
with tab3:
    st.header("Step 2: Product Recommendation")

    if not st.session_state.face_authenticated:
        st.error("Face authentication required first!")
        st.stop()

    st.success("Face authenticated! Get your product recommendation below.")

    # Input features for prediction
    st.subheader("Enter Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        social_platform = st.selectbox(
            'Social Media Platform',
            df['social_media_platform'].unique()  # type: ignore
        )

        engagement_score = st.slider(
            'Engagement Score',
            float(df['engagement_score'].min()),
            float(df['engagement_score'].max()),
            float(df['engagement_score'].mean())
        )

        purchase_interest = st.slider(
            'Purchase Interest Score',
            float(df['purchase_interest_score'].min()),
            float(df['purchase_interest_score'].max()),
            float(df['purchase_interest_score'].mean())
        )

    with col2:
        sentiment = st.selectbox(
            'Review Sentiment',
            df['review_sentiment'].unique() # type: ignore
        )

        purchase_amount = st.number_input(
            'Average Purchase Amount ($)',
            min_value=float(df['purchase_amount'].min()),
            max_value=float(df['purchase_amount'].max()),
            value=float(df['purchase_amount'].mean())
        )

        rating = st.slider(
            'Customer Rating',
            float(df['customer_rating'].min()),
            float(df['customer_rating'].max()),
            float(df['customer_rating'].mean())
        )

    # Create input dataframe
    input_data = pd.DataFrame({
            'engagement_score': [engagement_score],
            'purchase_interest_score': [purchase_interest],
            'purchase_amount': [purchase_amount],
            'customer_rating': [rating],
            'review_sentiment': [sentiment],
            'primary_platform': [social_platform]
    })

    # Show input data
    with st.expander("View Input Data"):
        st.dataframe(input_data, use_container_width=True) # type: ignore

    # Make prediction
    if st.button("Get Product Recommendation", type="primary"):
        try:

            # Predict
            prediction = models['product_model'].predict(input_data)
            prediction_proba = models['product_model'].predict_proba(input_data)
            # Decode prediction
            predicted_category = models['label_encoder'].inverse_transform(prediction)[0]

            # For random forest
            # predicted_category = prediction[0]

            # Store prediction in session state
            st.session_state.prediction = predicted_category
            st.session_state.prediction_proba = prediction_proba

            # Display results
            st.success(f"### Recommended Product: **{predicted_category}**")
            st.info("**Please proceed to Voice Confirmation tab to approve this recommendation**")

            # Show probabilities
            st.subheader("Prediction Confidence")
            proba_df = pd.DataFrame({
                'Product Category': models['label_encoder'].classes_, # type: ignore
                'Probability': prediction_proba[0]
            }).sort_values('Probability', ascending=False)

            st.dataframe(proba_df, use_container_width=True) # type: ignore
            st.bar_chart(proba_df.set_index('Product Category')) # type: ignore

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Debug: Check that your model was trained with the correct features")
# Tab 4: Voice Confirmation
with tab4:
    st.header("Step 3: Voice Confirmation")

    if not st.session_state.face_authenticated:
        st.error("Face authentication required first!")
        st.stop()

    if st.session_state.prediction is None:
        st.warning("Please get a product recommendation first (Tab 3)")
        st.stop()

    st.info(f"**Pending Recommendation:** {st.session_state.prediction}")
    st.write("Upload audio saying"
    " **'Yes, approve'** or **'Confirm transaction'** to confirm this recommendation")

    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'mp4'],
        key="audio_upload"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        # Simulated voice verification (replace with actual voice model)
        voice_choice = st.radio(
            "Simulate voice verification result:",
            ["Authorized Voice", "Unauthorized Voice"]
        )

        if st.button("Verify & Approve", type="primary"):
            if voice_choice == "Authorized Voice":
                st.session_state.voice_authenticated = True
                st.success("Voice verified! Transaction approved!")

                # Display final approved recommendation
                st.balloons()
                st.success(f"### APPROVED: {st.session_state.prediction}")
                # Show confidence
                proba_df = pd.DataFrame({
                    'Product Category': models['label_encoder'].classes_, # type: ignore
                    'Probability': st.session_state.prediction_proba[0] # type: ignore
                }).sort_values('Probability', ascending=False)

                st.subheader("Final Prediction Confidence")
                st.dataframe(proba_df, use_container_width=True) # type: ignore
                st.bar_chart(proba_df.set_index('Product Category')) # type: ignore

            else:
                st.error("Voice not recognized. Transaction denied.")
                st.session_state.voice_authenticated = False
                st.session_state.prediction = None
                st.session_state.prediction_proba = None
