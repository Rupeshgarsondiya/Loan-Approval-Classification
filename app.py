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

# --------------------
# CONFIGURATION
# --------------------
# Dynamic version selection - gets the latest version
def get_latest_version():
    """Get the latest version folder available"""
    base_path = "save_models"
    if not os.path.exists(base_path):
        return "version1"  # fallback
    
    versions = [d for d in os.listdir(base_path) if d.startswith("version") and os.path.isdir(os.path.join(base_path, d))]
    if not versions:
        return "version1"  # fallback
    
    # Sort versions numerically
    versions.sort(key=lambda x: int(x.replace("version", "")))
    return versions[-1]  # Return latest version

VERSION = get_latest_version()
MODEL_DIR = os.path.join("save_models", VERSION)
IMAGE_PATH = os.path.join("Loan_Approval.png")  # Relative path for portability

ALGORITHMS = {
    "Logistic Regression": "logisticregression.pkl",
    "Decision Tree": "decisiontree.pkl",
    "Random Forest": "randomforest.pkl",
    "KNN": "knn.pkl",
    "Neural Network": "neuralnetwork.h5"
}

# --------------------
# INFERENCE PREPROCESSING CLASS
# --------------------
class InferencePreprocessor:
    """
    Handles preprocessing for inference using saved preprocessors
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.preprocessor = None
        self.label_encoder = None
        self._load_preprocessors()
    
    def _load_preprocessors(self):
        """Load the fitted preprocessors from training"""
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
        """
        Preprocess input data using the fitted preprocessor
        
        Args:
            input_data (pd.DataFrame): Raw input data
            
        Returns:
            np.array: Transformed data ready for model prediction
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded. Cannot transform data.")
        
        try:
            # Transform the data using the fitted preprocessor
            transformed_data = self.preprocessor.transform(input_data)
            return transformed_data
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            return None
    
    def decode_prediction(self, prediction):
        """
        Decode numerical prediction back to original labels
        
        Args:
            prediction: Numerical prediction (0 or 1)
            
        Returns:
            str: Original label
        """
        if self.label_encoder is None:
            # Fallback mapping if label encoder not available
            return "Approved" if prediction == 1 else "Not Approved"
        
        try:
            decoded = self.label_encoder.inverse_transform([prediction])[0]
            return "Approved" if decoded == 1 else "Not Approved"
        except Exception as e:
            st.error(f"Error decoding prediction: {str(e)}")
            return "Approved" if prediction == 1 else "Not Approved"

# --------------------
# STREAMLIT PAGE SETTINGS
# --------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for scrollable inputs and static Submit button
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
        padding: 8px; 
        font-size: 16px; 
    }
    .stButton>button:hover { 
        background-color: #45a049; 
    }
    .stSelectbox, .stRadio, .stNumberInput { 
        margin-bottom: 8px; 
    }
    .stSelectbox div[data-baseweb="select"]>div { 
        padding: 4px; 
    }
    .stNumberInput input { 
        padding: 4px; 
    }
    .stRadio div[role="radiogroup"]>div { 
        padding: 4px; 
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

# --------------------
# INITIALIZE PREPROCESSOR
# --------------------
@st.cache_resource
def load_inference_preprocessor():
    """Load and cache the inference preprocessor"""
    return InferencePreprocessor(MODEL_DIR)

# Load preprocessor
inference_processor = load_inference_preprocessor()

# --------------------
# LAYOUT
# --------------------
st.markdown('<div class="main">', unsafe_allow_html=True)

# Version info
st.markdown(f'<div class="version-info"><b>Using Model Version:</b> {VERSION}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])  # Equal columns for balance

with col1:
    st.markdown('<div class="subheader">Input Parameters</div>', unsafe_allow_html=True)
    
    # Scrollable container for input fields
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        selected_algo = st.selectbox("Select Algorithm", list(ALGORITHMS.keys()), help="Choose the ML model")
        gender = st.radio("Gender", ["Male", "Female"], help="Select gender")
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"], help="Highest education level")
        home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"], help="Home ownership status")
        loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], help="Loan purpose")
        prev_default = st.radio("Previous Loan Default?", ["Yes", "No"], help="Any prior loan defaults?")
        
        person_age = st.number_input("Age", min_value=18, max_value=100, value=25, step=1, help="Age (18-100)", format="%d")
        person_income = st.number_input("Income", min_value=0, value=50000, step=1000, help="Annual income", format="%d")
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, value=3, step=1, help="Work experience", format="%d")
        loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000, step=100, help="Loan amount", format="%d")
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=10.0, step=0.1, help="Interest rate", format="%.2f")
        loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, value=0.2, step=0.01, help="Loan as % of income", format="%.2f")
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=5, step=1, help="Credit history years", format="%d")
        credit_score = st.number_input("Credit Score", min_value=0, max_value=850, value=650, step=1, help="Credit score (0-850)", format="%d")
        st.markdown('</div>', unsafe_allow_html=True)

    # Static Submit button
    st.markdown('<div class="submit-button">', unsafe_allow_html=True)
    submit_clicked = st.button("Submit", key="submit_button")
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

    # Image with error handling, using use_container_width
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, caption="Loan Approval Prediction", use_container_width=True)
    else:
        st.warning("Image not found. Using placeholder.")
        st.image("/home/rupesh-garsondiya/workstation/lab/Loan-Approval-Classification/Loan-Approval-Classification/Loan_Approval.png", caption="Loan Approval Prediction", wdith=1000)

    # Display model information if available
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

# --------------------
# PREDICTION
# --------------------
if submit_clicked:
    # Validate that preprocessor is loaded
    if inference_processor.preprocessor is None:
        st.error("❌ Cannot make predictions. Preprocessor not loaded properly.")
        st.stop()
    
    # Create input dataframe with exact column names and order as used in training
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
    
    # Display input data for debugging
    with st.expander("🔍 View Input Data"):
        st.write("**Raw Input Data:**")
        st.dataframe(input_data)
    
    # Preprocess the input data
    st.write("🔄 Preprocessing input data...")
    input_transformed = inference_processor.preprocess_input(input_data)
    
    if input_transformed is not None:
        # Display transformed data shape
        st.write(f"✅ Data transformed successfully. Shape: {input_transformed.shape}")
        
        # Load and use the selected model
        model_path = os.path.join(MODEL_DIR, ALGORITHMS[selected_algo])
        
        try:
            st.write(f"🤖 Loading {selected_algo} model...")
            
            if selected_algo == "Neural Network":
                model = tf.keras.models.load_model(model_path)
                prediction_prob = model.predict(input_transformed, verbose=0)[0][0]
                prediction = 1 if prediction_prob >= 0.5 else 0
                
                # Display probability and prediction
                st.write(f"**Prediction Probability**: {prediction_prob:.4f}")
                pred_label = "Approved" if prediction == 1 else "Not Approved"
                
            else:
                model = joblib.load(model_path)
                prediction = model.predict(input_transformed)[0]
                
                # Get prediction probability if available
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(input_transformed)[0]
                    st.write(f"**Prediction Probabilities**: Not Approved: {prob[0]:.4f}, Approved: {prob[1]:.4f}")
                
                pred_label = "Approved" if prediction == 1 else "Not Approved"
            
            # Display final result
            if pred_label == "Approved":
                st.success(f"🎉 **Prediction: {pred_label}**")
            else:
                st.error(f"❌ **Prediction: {pred_label}**")
                
            st.write(f"**Algorithm Used**: {selected_algo}")
            st.write(f"**Model Version**: {VERSION}")
            
        except FileNotFoundError:
            st.error(f"❌ Model file {model_path} not found. Please ensure the model is saved in {MODEL_DIR}.")
            st.write("**Available files in model directory:**")
            if os.path.exists(MODEL_DIR):
                files = os.listdir(MODEL_DIR)
                for file in files:
                    st.write(f"- {file}")
            else:
                st.write("Model directory does not exist")
                
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")
            st.write("**Error Details:**")
            st.exception(e)
    else:
        st.error("❌ Failed to preprocess input data. Cannot make prediction.")

# --------------------
# FOOTER
# --------------------
st.markdown(
    '<div class="footer">© 2025 Rupesh Garsondiya. All Rights Reserved.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)