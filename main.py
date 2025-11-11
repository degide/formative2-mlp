"""
Formative 2 - Multimodal Authentication & Product Recommendation System
A Streamlit app implementing:
1. Face Recognition Authentication
2. Voice Verification Authentication  
3. Product Recommendation Model
"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

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
def load_models():
    """Load all trained models"""
    models = {}
    try:
        # Load model info to determine best model
        with open(MODEL_DIR / "best_model_info.json", 'r') as f:
            models['model_info'] = json.load(f)

        # Try to load XGBoost (best model) first
        best_model_name = models['model_info'].get('best_model_name', 'Random Forest')

        if best_model_name == 'XGBoost':
            try:
                models['product_model'] = joblib.load(MODEL_DIR / "product_recommender_xgb.joblib")
                models['model_used'] = 'XGBoost'
            except Exception as xgb_error:
                st.warning(f"⚠️ XGBoost model failed to load: {str(xgb_error)[:100]}... Using Random Forest instead.")
            models['product_model'] = joblib.load(MODEL_DIR / "product_recommender_rf.joblib")
            models['model_used'] = 'Random Forest (fallback)'
        else:
            models['product_model'] = joblib.load(MODEL_DIR / "product_recommender_rf.joblib")
            models['model_used'] = 'Random Forest'

        models['label_encoder'] = joblib.load(MODEL_DIR / "label_encoder.joblib")

        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return None

@st.cache_data
def load_data():
    """Load customer data"""
    try:
        df = pd.read_csv(PROCESSED_DIR / "merged_customer_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize session state
if 'face_authenticated' not in st.session_state:
    st.session_state.face_authenticated = False
if 'voice_authenticated' not in st.session_state:
    st.session_state.voice_authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

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
        st.success("✅ Face Authenticated")
    else:
        st.error("❌ Face Not Authenticated")

    # Voice authentication status
    if st.session_state.voice_authenticated:
        st.success("✅ Voice Verified")
    else:
        st.error("❌ Voice Not Verified")

    st.divider()

    # Model info
    st.subheader("🤖 Model Information")
    st.info(f"**Active Model:** {models['model_used']}")

    st.divider()

    # Reset button
    if st.button("🔄 Reset Authentication"):
        st.session_state.face_authenticated = False
        st.session_state.voice_authenticated = False
        st.session_state.current_user = None
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "👤 Face Authentication",
      "🎤 Voice Verification", "🎯 Product Prediction"])

# Tab 1: Data Overview
with tab1:
    st.header("Customer Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df['customer_id_new'].unique()))
    with col2:
        st.metric("Total Transactions", len(df))
    with col3:
        st.metric("Product Categories", len(df['product_category'].unique()))
    with col4:
        st.metric("Avg Purchase Amount", f"${df['purchase_amount'].mean():.2f}")


    with st.expander("View Full Dataset"):
        st.dataframe(df, use_container_width=True)

    with st.expander("View Data Sample"):
        st.dataframe(df.describe(), use_container_width=True)

    product_counts = df['product_category'].value_counts()
    with st.expander("View Category Counts"):
        st.bar_chart(product_counts)

# Tab 2: Face Authentication
with tab2:
    st.header("👤 Step 1: Face Recognition Authentication")

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
                    st.success("✅ Face recognized! You may proceed to voice verification.")
                else:
                    st.session_state.face_authenticated = False
                    st.error("❌ Face not recognized. Access denied.")
                st.rerun()

    if not st.session_state.face_authenticated:
        st.warning("⚠️ Please authenticate your face to proceed to voice verification")

# Tab 3: Voice Verification
with tab3:
    st.header("🎤 Step 2: Voice Verification")

    if not st.session_state.face_authenticated:
        st.error("❌ Face authentication required first!")
        st.stop()

    st.write("Upload an audio sample saying 'Yes, approve' or 'Confirm transaction'")

    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg'],
        key="audio_upload"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        # Simulated voice verification (replace with actual voice model)
        voice_choice = st.radio(
            "Simulate voice verification result:",
            ["Authorized Voice", "Unauthorized Voice"]
        )

        if st.button("Verify Voice", type="primary"):
            if voice_choice == "Authorized Voice":
                st.session_state.voice_authenticated = True
                st.success("✅ Voice verified! You may now access product recommendations.")
            else:
                st.session_state.voice_authenticated = False
                st.error("❌ Voice not recognized. Access denied.")
            st.rerun()

    if not st.session_state.voice_authenticated:
        st.warning("⚠️ Please verify your voice to access product recommendations")

# Tab 4: Product Prediction
with tab4:
    st.header("🎯 Step 3: Product Recommendation")

    if not st.session_state.face_authenticated:
        st.error("❌ Face authentication required!")
        st.stop()

    if not st.session_state.voice_authenticated:
        st.error("❌ Voice verification required!")
        st.stop()

    st.success("✅ All authentication steps completed! You can now get product recommendations.")

    # Input features for prediction
    st.subheader("Enter Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        social_platform = st.selectbox(
            'Social Media Platform',
            df['social_media_platform'].unique()
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
            df['review_sentiment'].unique()
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
        'social_media_platform': [social_platform],
        'engagement_score': [engagement_score],
        'purchase_interest_score': [purchase_interest],
        'review_sentiment': [sentiment],
        'purchase_amount': [purchase_amount],
        'customer_rating': [rating]
    })

    # Show input data
    with st.expander("View Input Data"):
        st.dataframe(input_data, use_container_width=True)

    # Make prediction
    if st.button("🎯 Get Product Recommendation", type="primary"):
        try:
            # Prepare data (encode categorical variables)
            input_encoded = pd.get_dummies(input_data, columns=['social_media_platform', 'review_sentiment'])

            # Align columns with training data
            model_features = models['product_model'].feature_names_in_
            for col in model_features:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_features]

            # Predict
            prediction = models['product_model'].predict(input_encoded)
            prediction_proba = models['product_model'].predict_proba(input_encoded)

            # Decode prediction
            predicted_category = models['label_encoder'].inverse_transform(prediction)[0]

            # Display results
            st.success(f"### 🎁 Recommended Product Category: **{predicted_category}**")

            # Show probabilities
            st.subheader("Prediction Confidence")
            proba_df = pd.DataFrame({
                'Product Category': models['label_encoder'].classes_,
                'Probability': prediction_proba[0]
            }).sort_values('Probability', ascending=False)

            st.dataframe(proba_df, use_container_width=True)
            st.bar_chart(proba_df.set_index('Product Category'))

            # Model performance info
            with st.expander("Model Performance"):
                st.json(models['model_info'])

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Debug: Check that your model was trained with the correct features")
