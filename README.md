# Formative 2: Multimodal Data Preprocessing Assignment - MLP

## Assignment Overview

**Try the app:** [https://my-recommendation.streamlit.app/](https://my-recommendation.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://my-recommendation.streamlit.app/)

This project implements a **User Identity and Product Recommendation System** that authenticates users through **facial recognition** and **voice validation** before providing personalized product recommendations. The system enforces multiple security checkpoints, if any authentication step fails, access to recommendations is denied.

**System Flow:**
1. User attempts to access the product prediction model.
2. If the face is recognized, the user can request a product prediction.
3. The prediction must be confirmed through voice verification.
4. The system uses pre-trained models to check:
   - Face matches a known user (facial recognition)
   - Voice matches an approved sample (voiceprint verification)
   - Only then is the prediction displayed.


## Project Structure

```sh
formative2-mlp/
    ├── check.py
    ├── data
    │   ├── notebooks # Jupyter notebooks
    │   ├── processed # Processed datasets
    │   ├── raw # Raw datasets
    │   ├── requirements.txt
    │   └── scripts # Scripts
    ├── LICENSE
    ├── main.py
    ├── models # Exported models
    ├── README.md
    ├── requirements.txt
    └── utils # Utilities
```


---

## Tasks & Implementation

### 1. Data Merge
- Merged `customer_social_profiles` and `customer_transactions` datasets.
- Engineered features from both sources.
- Saved as [`data/processed/merged_customer_data.csv`](data/processed/merged_customer_data.csv).

### 2. Image Data Collection and Processing
- Each member submitted at least 3 facial images (neutral, smiling, surprised).
- Images were augmented (rotation, flipping, grayscale).
- Extracted features (embeddings, histograms) and saved to `image_features.csv`.

### 3. Sound Data Collection and Processing
- Each member recorded at least 2 audio samples (e.g., “Yes, approve”, “Confirm transaction”).
- Audio samples were augmented (pitch shift, time stretch, background noise).
- Extracted features (MFCCs, spectral roll-off, energy) and saved to `audio_features.csv`.

### 4. Model Creation
- **Facial Recognition Model:** Classifies users based on facial features.
- **Voiceprint Verification Model:** Authenticates users via voice features.
- **Product Recommendation Model:** Predicts product category using merged features.
- Models used: Random Forest, Logistic Regression, XGBoost, Keras NN.
- Evaluated using Accuracy, F1-Score, and Loss.

### 5. System Demonstration
- Simulated unauthorized attempts (image/audio).
- Simulated full transaction: face authentication → product prediction → voice confirmation.
- Implemented as a Streamlit mini-app (`main.py`).

---

## Deliverables

- **Datasets:** Raw, processed, and merged datasets.
- **Feature Files:** `image_features.csv`, `audio_features.csv`.
- **Scripts:** For data merging, feature extraction, model training, and prediction.
- **Jupyter Notebooks:** For data exploration, training, and evaluation.
- **System Simulation:** Video link (see report).
- **Report:** Detailed approach, results, and team contributions.

---

## How to Run

1. **Install dependencies:**
   ```sh
    pip install -r requirements.txt
   ```
2. **Run the Streamlit app:**
    ```sh
    streamlit run main.py
    ```
3. **Explore Notebooks:**
    See data/notebooks/ for step-by-step data processing and model training.

## Licence

MIT