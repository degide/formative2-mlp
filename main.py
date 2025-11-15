"""
Formative 2 - Multimodal Authentication & Product Recommendation System
A Streamlit app implementing:
1. Face Recognition Authentication
2. Voice Verification Authentication  
3. Product Recommendation Model
"""
import os
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
from utils.load_models import load_all_models, load_data, extract_audio_features, predict_speaker # type: ignore
sys.path.insert(0, str(Path(__file__).parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TensorFlow early and configure it
import tensorflow as tf # type: ignore
tf.config.set_visible_devices([], 'GPU')  # Disable GPU/Metal # type: ignore

# Page configuration
st.set_page_config(
    page_title="Recomm",
    page_icon="🛍️",
    layout="wide"
)

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

# ==================== LOAD DATA AND MODELS ====================

# Load data and models
df = load_data()
with st.spinner("Loading models..."):
    models = load_all_models()  # type: ignore
    st.toast("Data and models loaded successfully!", icon="✅")

if df is None or models is None:
    st.stop()

# Initialize session state
if 'face_authenticated' not in st.session_state:
    st.session_state.face_authenticated = False
if 'voice_authenticated' not in st.session_state:
    st.session_state.voice_authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = "JD"
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None
if 'voice_attempts' not in st.session_state:
    st.session_state.voice_attempts = 0

# Main title
st.title('🛍️ Product Recommendation System')
st.info('🔐 A sequential authentication system using' \
' facial recognition and voice validation before providing personalized product recommendations')

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

    # Product model status
    product_status = models.get('model_used') # type: ignore
    st.info(f"**Product Model:** {product_status}")

    # Face model status
    face_status = "Loaded" if models.get('face_model') is not None else "Simulation" # type: ignore
    st.info(f"**Face Model:** {face_status}")

    # Voice model status
    voice_status = "Loaded" if models.get('voice_model') is not None else "Simulation" # type: ignore
    st.info(f"**Voice Model:** {voice_status}")

    st.divider()

    # Reset button
    if st.button("Reset Authentication"):
        st.session_state.face_authenticated = False
        st.session_state.voice_authenticated = False
        st.session_state.current_user = None
        st.session_state.prediction = None
        st.session_state.prediction_proba = None
        st.session_state.voice_attempts = 0
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

    product_counts = df['product_category'].value_counts()
    with st.expander("View Category Counts"):
        st.bar_chart(product_counts) # type: ignore

    # Replace the "My models" expander section (around line 140) with this:

    with st.expander("System Models Overview"):

        # ==================== PRODUCT RECOMMENDATION MODEL ====================
        with st.expander("Product Recommendation Model Details"):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Model Type", models.get('model_used', 'N/A'))
                st.metric("Status", "Loaded" if models.get('product_model') else "Not Loaded")

            with col2:
                if models.get('label_encoder'):
                    num_categories = len(models['label_encoder'].classes_)
                    st.metric("Product Categories", num_categories)

                if models.get('product_model') and hasattr(models['product_model'], 'feature_names_in_'):
                    num_features = len(models['product_model'].feature_names_in_)
                    st.metric("Input Features", num_features)

            # Product Model Features
            if models.get('product_model') and hasattr(models['product_model'], 'feature_names_in_'):
                with st.expander("View Product Model Features"):
                    features = models['product_model'].feature_names_in_
                    st.write("**Required Input Features:**")

                    # Display features in a nice table
                    features_df = pd.DataFrame({
                        'Feature Name': features
                    })
                    st.dataframe(features_df, use_container_width=True, hide_index=False) #type: ignore

            # Product Categories
            if models.get('label_encoder'):
                with st.expander("View Product Categories"):
                    categories = models['label_encoder'].classes_
                    st.write("**Predictable Product Categories:**")

                    # Display categories in a nice table with indices
                    categories_df = pd.DataFrame({
                        'Category Name': categories
                    })
                    st.dataframe(categories_df, use_container_width=True, hide_index=False) #type: ignore

        # ==================== FACE RECOGNITION MODEL ====================
        with st.expander("Face Recognition Model Details"):

            col1, col2 = st.columns(2)

            with col1:
                if models.get('face_model'):
                    st.metric("Status", "Loaded")
                    st.metric("Model Type", type(models['face_model']).__name__)
                else:
                    st.metric("Status", "Simulation Mode")
                    if models.get('face_error'):
                        st.caption(f"Reason: {models['face_error']}")

            with col2:
                if models.get('face_model'):
                    # Show additional face model info if available
                    if hasattr(models['face_model'], 'n_features_in_'):
                        st.metric("Input Features", models['face_model'].n_features_in_)
                    if hasattr(models['face_model'], 'classes_'):
                        st.metric("Recognized Users", len(models['face_model'].classes_))

            # Face Model Details
            if models.get('face_model'):
                with st.expander("View Face Model Details"):
                    st.write("**Model Information:**")

                    # Dynamically show all available attributes
                    face_info = {}
                    for attr in ['n_features_in_', 'classes_', 'feature_names_in_']:
                        if hasattr(models['face_model'], attr):
                            face_info[attr] = getattr(models['face_model'], attr)

                    if face_info:
                        st.json(face_info) #type: ignore
                    else:
                        st.info("Model loaded but no additional details available")

        # ==================== VOICE VERIFICATION MODEL ====================
        with st.expander("Voice Verification Model Details"):

            col1, col2 = st.columns(2)

            with col1:
                if models.get('voice_model'):
                    st.metric("Status", "Loaded")
                    st.metric("Model Type", type(models['voice_model']).__name__)
                else:
                    st.metric("Status", "Simulation Mode")
                    if models.get('voice_error'):
                        st.caption(f"Reason: {models['voice_error']}")

            with col2:
                if models.get('voice_model'):
                    # Show additional voice model info if available
                    if hasattr(models['voice_model'], 'n_features_in_'):
                        st.metric("Input Features", models['voice_model'].n_features_in_)
                    if hasattr(models['voice_model'], 'classes_'):
                        st.metric("Voice Profiles", len(models['voice_model'].classes_))

            # Voice Model Details
            if models.get('voice_model'):
                with st.expander("View Voice Model Details"):
                    st.write("**Model Information:**")

                    # Dynamically show all available attributes
                    voice_info = {}
                    for attr in ['n_features_in_', 'classes_', 'feature_names_in_']:
                        if hasattr(models['voice_model'], attr):
                            voice_info[attr] = getattr(models['voice_model'], attr)

                    if voice_info:
                        st.json(voice_info) #type: ignore
                    else:
                        st.info("Model loaded but no additional details available")

        # ==================== ERROR SUMMARY ====================
        if models.get('errors') or models.get('face_error') or models.get('voice_error'):
            st.markdown("### Loading Warnings/Errors")

            all_errors = []

            # Product model errors
            if models.get('errors'):
                all_errors.extend(models['errors']) # type: ignore

            # Face model errors
            if models.get('face_error'):
                all_errors.append(f"Face Model: {models['face_error']}") # type: ignore

            # Voice model errors
            if models.get('voice_error'):
                all_errors.append(f"Voice Model: {models['voice_error']}") # type: ignore
            for i, error in enumerate(all_errors, 1): # type: ignore
                st.warning(f"{i}. {error}")

# Tab 2: Face Authentication
with tab2:
    st.header("Sterp 1: Face Recognition Authentication")

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
            st.write("### Authentication")

            # Check if face model is available
            if models.get('face_model') is not None and models.get('face_class_names') is not None:
                # REAL FACE RECOGNITION MODEL
                st.info("Using AI face recognition model")

                if st.button("Authenticate Face", type="primary"):
                    with st.spinner("Analyzing face..."):
                        # Import preprocessing function
                        import numpy as np
                        from utils.load_models import preprocess_face_image

                        # Preprocess image
                        processed_image = preprocess_face_image(uploaded_image)

                        if processed_image is not None:
                            import numpy as np
                            try:
                                # Make prediction
                                predictions = models['face_model'].predict(processed_image, verbose=0)
                                predicted_class_idx = np.argmax(predictions[0])
                                confidence = predictions[0][predicted_class_idx]
                                predicted_name = models['face_class_names'][predicted_class_idx]

                                # Show all predictions
                                st.write("**Recognition Results:**")
                                for idx, class_name in enumerate(models['face_class_names']):
                                    prob = predictions[0][idx]
                                    st.write(f"- {class_name}: {prob:.2%}")

                                # Decision threshold (adjust as needed)
                                CONFIDENCE_THRESHOLD = 0.80  # 80% confidence

                                if confidence >= CONFIDENCE_THRESHOLD:
                                    st.session_state.face_authenticated = True
                                    st.session_state.current_user = predicted_name
                                    st.success(f"Face recognized as **{predicted_name}** with {confidence:.2%} confidence!")
                                    st.success("Proceed to Product Recommendation tab.")
                                    st.balloons()
                                else:
                                    st.session_state.face_authenticated = False
                                    st.error(f"Unrecognized face (Confidence: {confidence:.2%} < {CONFIDENCE_THRESHOLD:.0%})")
                                    st.warning("Access denied.")

                            except Exception as e:
                                st.error(f"Face recognition error: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                        else:
                            st.error("Failed to process image. Please upload a valid image file.")

            else:
                # SIMULATION MODE (fallback when model not available)
                st.warning("Face model not available - using simulation mode")

                auth_choice = st.radio(
                    "Simulate authentication result:",
                    ["Authorized User", "Unauthorized User"]
                )

                if st.button("Authenticate Face", type="primary"):
                    if auth_choice == "Authorized User":
                        st.session_state.face_authenticated = True
                        st.session_state.current_user = "JD"  # Simulated user ID
                        st.success("Face recognized! (Simulation)")
                        st.success("Proceed to Product Recommendation tab.")
                    else:
                        st.session_state.face_authenticated = False
                        st.error("Face not recognized. (Simulation)")
                        st.error("Access denied.")

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
            prediction = models['product_model'].predict(input_data) # type: ignore
            prediction_proba = models['product_model'].predict_proba(input_data) # type: ignore
            # Decode prediction
            predicted_category = models['label_encoder'].inverse_transform(prediction)[0] # type: ignore

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
    st.write("Upload audio saying 'Yes, approve' or 'Confirm transaction' to confirm this recommendation")


    uploaded_audio = st.file_uploader(
        "Choose an audio file (WAV recommended)",
        type=['wav', 'mp3', 'ogg', 'mp4'],
        key="audio_upload"
    )

    if uploaded_audio is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.audio(uploaded_audio)
            st.caption(f"File: {uploaded_audio.name}")
            st.caption(f"Size: {uploaded_audio.size / 1024:.2f} KB")

        with col2:
            st.write("### Voice Verification")

            if models.get('voice_model') is not None:
                st.info("Using AI voice verification")

                # Configuration
                CONFIDENCE_THRESHOLD = 0.70   # 70% minimum confidence
                MARGIN_THRESHOLD = 0.15       # 15% gap between 1st and 2nd
                MAX_ATTEMPTS = 3

                if st.button("Verify & Approve", type="primary"):
                    with st.spinner("Analyzing voice patterns..."):
                        # Extract audio features
                        audio_features = extract_audio_features(uploaded_audio)

                        if audio_features is None:
                            st.error("Failed to extract audio features")
                            st.error("**Possible reasons:**")
                            st.error("- Audio file is corrupted or in unsupported format")
                            st.error("- File is MP4 disguised as WAV")
                            st.error("- Audio quality is too poor")
                            st.info("**Tip:** Use real WAV files from `sample_audio` folder")
                            st.stop()

                        # Predict speaker
                        speaker, confidence, all_probs = predict_speaker(
                            models['voice_model'],
                            models['feature_names'],
                            models['voice_class_names'],
                            audio_features
                        )

                        if speaker is None or confidence is None or all_probs is None:
                            st.error("Voice recognition failed")
                            st.error("The model could not process this audio file.")
                            st.info("Try uploading a clearer audio recording")
                            st.stop()

                        # Calculate margin between top 2 predictions
                        sorted_probs = sorted(enumerate(all_probs), key=lambda x: x[1], reverse=True)
                        first_idx, first_prob = sorted_probs[0]
                        second_idx, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else (0, 0)

                        predicted_speaker = models['voice_class_names'][first_idx]
                        margin = first_prob - second_prob

                        # Show detailed results
                        st.write("---")
                        st.write("### Voice Analysis Results")

                        # All predictions
                        with st.expander("View All Predictions"):
                            for person, prob in zip(models['voice_class_names'], all_probs):
                                st.write(f"- **{person}**: {prob:.2%}")

                        # Key metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Predicted Speaker", predicted_speaker)
                            st.metric("Confidence", f"{first_prob:.2%}")
                        with col_b:
                            st.metric("Expected User", st.session_state.current_user)
                            st.metric("Confidence Margin", f"{margin:.2%}")

                        st.write("---")
                        st.write("### Security Checks")

                        # Multi-condition validation
                        conditions_met = []
                        all_passed = True

                        # Condition 1: Speaker matches
                        if predicted_speaker == st.session_state.current_user:
                            conditions_met.append("**Speaker Match:** Voice matches expected user")
                        else:
                            conditions_met.append(f"**Speaker Mismatch:** Expected `{st.session_state.current_user}`, got `{predicted_speaker}`")
                            all_passed = False

                        # Condition 2: Confidence threshold
                        if first_prob >= CONFIDENCE_THRESHOLD:
                            conditions_met.append(f"**Confidence:** {first_prob:.2%} ≥ {CONFIDENCE_THRESHOLD:.0%} threshold")
                        else:
                            conditions_met.append(f"**Low Confidence:** {first_prob:.2%} < {CONFIDENCE_THRESHOLD:.0%} threshold")
                            all_passed = False

                        # Condition 3: Clear winner (margin check)
                        if margin >= MARGIN_THRESHOLD:
                            conditions_met.append(f"**Clear Winner:** {margin:.2%} margin ≥ {MARGIN_THRESHOLD:.0%}")
                        else:
                            conditions_met.append(f"**Ambiguous:** {margin:.2%} margin < {MARGIN_THRESHOLD:.0%} (voices too similar)")
                            # Don't fail on margin alone, just warn
                            # all_passed = False

                        # Display all conditions
                        for condition in conditions_met:
                            if "Yes" in condition:
                                st.success(condition)
                            elif "No" in condition:
                                st.error(condition)
                            else:
                                st.warning(condition)

                        st.write("---")

                        # Final decision
                        if all_passed and predicted_speaker == st.session_state.current_user and first_prob >= CONFIDENCE_THRESHOLD:
                            st.session_state.voice_authenticated = True
                            st.session_state.voice_attempts = 0

                            st.success("### VOICE VERIFICATION SUCCESSFUL!")
                            st.success(f"Voice verified as **{speaker}** with {first_prob:.2%} confidence")
                            st.success("### TRANSACTION APPROVED!")
                            st.success(f"**Recommended Product:** {st.session_state.prediction}")
                            st.balloons()

                        else:
                            # Failed verification
                            st.session_state.voice_attempts += 1
                            remaining = MAX_ATTEMPTS - st.session_state.voice_attempts

                            with st.expander("View Failure Details"):
                                st.error("### VOICE VERIFICATION FAILED")
                                st.error("Transaction denied - security checks not passed")

                                if remaining > 0:
                                    st.warning(f"Attempts Remaining: {remaining}/{MAX_ATTEMPTS}")

                                    st.info("### Tips for Better Results:")
                                    st.info("- Speak clearly in a quiet environment")
                                    st.info("- Use a good quality microphone")
                                    st.info("- Record for at least 2-3 seconds")
                                    st.info("- Ensure audio file is in real WAV format")
                                    st.info("- Try files from `sample_audio` folder")
                                else:
                                    st.error("### MAXIMUM ATTEMPTS REACHED")
                                    st.error("Transaction permanently denied.")
                                    st.error("Please reset authentication and try again.")
                                    st.session_state.voice_attempts = 0

            else:
                # SIMULATION MODE
                st.warning("Voice model not available - using simulation mode")

                voice_choice = st.radio(
                    "Simulate voice verification:",
                    ["Authorized Voice", "Unauthorized Voice"]
                )

                if st.button("Verify & Approve", type="primary"):
                    if voice_choice == "Authorized Voice":
                        st.session_state.voice_authenticated = True
                        st.session_state.voice_attempts = 0
                        st.success("Voice verified! (Simulation)")
                        st.success(f"### APPROVED: {st.session_state.prediction}")
                        st.balloons()
                    else:
                        st.session_state.voice_authenticated = False
                        st.session_state.voice_attempts += 1
                        st.error("Voice not recognized. (Simulation)")
                        st.error("Transaction denied.")

    if not st.session_state.voice_authenticated:
        st.info("⏳ Waiting for voice confirmation...")
