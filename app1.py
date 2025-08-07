# app.py
"""
Author  : Rupesh Garsondiya
GitHub  : @Rupeshgarsondiya
Organization : L.J University
"""

import os
import joblib
import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import json
from src.scripts.preprocess_data import Preprocess

def get_latest_version():
    base_path = "save_models"
    if not os.path.exists(base_path):
        return "version1"
    versions = [d for d in os.listdir(base_path) if d.startswith("version") and os.path.isdir(os.path.join(base_path, d))]
    if not versions:
        return "version1"
    versions.sort(key=lambda x: int(x.replace("version", "")))
    return versions[-1]

VERSION = get_latest_version()
MODEL_DIR = os.path.join("save_models", VERSION)
IMAGE_PATH = os.path.join("Loan_Approval.png")

ALGORITHMS = {
    "Logistic Regression": "logisticregression.pkl",
    "Decision Tree": "decisiontree.pkl",
    "Random Forest": "randomforest.pkl",
    "KNN": "knn.pkl",
    "Neural Network": "neuralnetwork.h5"
}

class InferencePreprocessor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.preprocessor = None
        self.label_encoder = None
        self._load_preprocessors()

    def _load_preprocessors(self):
        try:
            preprocessor_path = os.path.join(self.model_dir, "preprocessor.pkl")
            label_encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")

            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                st.success(f"✅ Loaded preprocessor from {VERSION}")
            else:
                st.error(f"❌ Preprocessor not found in {preprocessor_path}")

            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                st.success(f"✅ Loaded label encoder from {VERSION}")
            else:
                st.error(f"❌ Label encoder not found in {label_encoder_path}")

        except Exception as e:
            st.error(f"Error loading preprocessors: {str(e)}")

    def preprocess_input(self, input_data):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Cannot transform data.")
        try:
            transformed_data = self.preprocessor.transform(input_data)
            return transformed_data
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            return None

    def decode_prediction(self, prediction):
        if self.label_encoder is None:
            return "Approved" if prediction == 1 else "Not Approved"
        try:
            decoded = self.label_encoder.inverse_transform([prediction])[0]
            return "Approved" if decoded == 1 else "Not Approved"
        except Exception as e:
            st.error(f"Error decoding prediction: {str(e)}")
            return "Approved" if prediction == 1 else "Not Approved"

st.set_page_config(page_title="Loan Approval Prediction", layout="wide", initial_sidebar_state="collapsed")





# CUSTOM CSS
st.markdown("""
    <style>
    .main { 
        padding: 10px; 
        max-height: 100vh; 
    }
    .input-container { 
        max-height: 60vh; 
        overflow-y: auto; 
        padding-right: 10px; 
    }
    .submit-button { 
        position: sticky; 
        bottom: 0; 
        background: #ffffff; 
        padding-top: 10px; 
    }
    .stButton>button { 
        width: 100%; 
        background-color: #4CAF50; 
        color: white; 
        padding: 10px; 
        font-size: 16px; 
    }
    .stButton>button:hover { 
        background-color: #45a049; 
    }
    .stSelectbox label, .stRadio label, .stNumberInput label { 
        font-weight: 600;
        font-size: 15px;
        color: #333;
    }
    .stSelectbox div[data-baseweb="select"] {
        font-size: 15px !important;
        padding: 8px !important;
        line-height: 1.6 !important;
    }
    .stSelectbox div[data-baseweb="select"] div[role="combobox"] {
        padding: 8px 12px !important;
    }
    .stNumberInput input, .stRadio div[role="radiogroup"]>div {
        padding: 6px 10px !important;
        font-size: 15px !important;
    }
    .title { 
        font-size: 2em; 
        text-align: center; 
        color: #333; 
        margin-bottom: 10px; 
    }
    .subheader { 
        font-size: 1.2em; 
        color: #555; 
        margin-bottom: 8px; 
    }
    .footer { 
        text-align: center; 
        color: #777; 
        margin-top: 15px; 
        font-size: 0.9em; 
    }
    .image-container { 
        display: flex; 
        justify-content: center; 
    }
    .version-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_inference_preprocessor():
    return InferencePreprocessor(MODEL_DIR)

inference_processor = load_inference_preprocessor()

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown(f'<div class="version-info"><b>Using Model Version:</b> {VERSION}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.3, 2])  # ✅ wider left column

with col1:
    st.markdown('<div class="subheader">Input Parameters</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        selected_algo = st.selectbox("Select Algorithm", list(ALGORITHMS.keys()))
        gender = st.radio("Gender", ["Male", "Female"])
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        prev_default = st.radio("Previous Loan Default?", ["Yes", "No"])

        person_age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
        person_income = st.number_input("Income", min_value=0, value=50000, step=1000)
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=3, step=1)
        loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000, step=100)
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.0, step=0.1)
        loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, value=0.2, step=0.01)
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5, step=1)
        credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=650, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="submit-button">', unsafe_allow_html=True)
    submit_clicked = st.button("Submit")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="title">Loan Approval Prediction</div>', unsafe_allow_html=True)
    st.markdown("### About This Project")
    st.markdown(f"""
    - A **Streamlit web app** to predict loan approval based on user inputs.
    - Supports multiple algorithms: Logistic Regression, Decision Tree, Random Forest, KNN, and Neural Network.
    - Trained on a comprehensive **Loan Data** dataset.
    - **Current Version**: {VERSION}
    """)

    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, caption="Loan Approval Prediction", use_container_width=False,width = 1000)
    else:
        st.warning("Image not found.")
        st.image("/home/rupesh-garsondiya/workstation/lab/Loan-Approval-Classification/Loan-Approval-Classification/Loan_Approval.png", caption="Loan Approval Prediction")

    try:
        summary_path = os.path.join(MODEL_DIR, "training_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            st.markdown("### Model Information")
            st.write(f"**Training Date**: {summary.get('timestamp', 'N/A')}")
            st.write(f"**Dataset**: {summary['dataset']['train_shape'][0]} samples, {summary['dataset']['train_shape'][1]} features")
    except Exception as e:
        st.write("Model information not available")

    st.markdown("[GitHub](https://github.com/Rupeshgarsondiya) | [LinkedIn](https://linkedin.com/in/rupeshgarsondiya)", unsafe_allow_html=True)

if submit_clicked:
    if inference_processor.preprocessor is None:
        st.error("❌ Cannot make predictions. Preprocessor not loaded properly.")
        st.stop()

    input_data = pd.DataFrame([[
        person_age, gender, education, person_income, person_emp_exp,
        home_ownership, loan_amnt, loan_intent, loan_int_rate,
        loan_percent_income, cb_person_cred_hist_length, credit_score,
        "Y" if prev_default == "Yes" else "N"
    ]], columns=[
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ])

    with st.expander("🔍 View Input Data"):
        st.dataframe(input_data)

    st.write("🔄 Preprocessing input data...")
    input_transformed = inference_processor.preprocess_input(input_data)

    if input_transformed is not None:
        st.write(f"✅ Data transformed successfully. Shape: {input_transformed.shape}")
        model_path = os.path.join(MODEL_DIR, ALGORITHMS[selected_algo])

        try:
            st.write(f"🤖 Loading {selected_algo} model...")
            if selected_algo == "Neural Network":
                model = tf.keras.models.load_model(model_path)
                prediction_prob = model.predict(input_transformed, verbose=0)[0][0]
                prediction = 1 if prediction_prob >= 0.5 else 0
                st.write(f"**Prediction Probability**: {prediction_prob:.4f}")
                pred_label = "Approved" if prediction == 1 else "Not Approved"
            else:
                model = joblib.load(model_path)
                prediction = model.predict(input_transformed)[0]
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(input_transformed)[0]
                    st.write(f"**Prediction Probabilities**: Not Approved: {prob[0]:.4f}, Approved: {prob[1]:.4f}")
                pred_label = "Approved" if prediction == 1 else "Not Approved"

            if pred_label == "Approved":
                st.success(f"🎉 **Prediction: {pred_label}**")
            else:
                st.error(f"❌ **Prediction: {pred_label}**")

            st.write(f"**Algorithm Used**: {selected_algo}")
            st.write(f"**Model Version**: {VERSION}")

        except FileNotFoundError:
            st.error(f"❌ Model file {model_path} not found.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.error("❌ Failed to preprocess input data.")

st.markdown('<div class="footer">© 2025 Rupesh Garsondiya. All Rights Reserved.</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
