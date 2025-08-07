"""
Name         : Rupesh Garsondiya
github       : @Rupeshgarsondiya
Organization : L.J University
"""

import os
import pandas as pd
import joblib  # Add missing import
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


class Preprocess:
    
    def __init__(self):
        self.cwd = os.getcwd()
        self.preprocessor = None
        self.label_encoder = None
        self.version_folder = self._get_next_version_folder()
    
    def _get_next_version_folder(self):
        """
        Get the next available version folder name
        
        return:
        - str: next version folder path (e.g., 'save_models/version1', 'save_models/version2')
        """
        base_path = os.path.join(self.cwd, 'save_models')
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        version_num = 1
        while True:
            version_folder = os.path.join(base_path, f'version{version_num}')
            if not os.path.exists(version_folder):
                return version_folder
            version_num += 1

    def load_data(self):  # Remove @staticmethod and self parameter issue
        """
        This function loads the data from raw data folder

        params
        -(None)

        return
        -data (pandas DataFrame)
        """
        data = pd.read_csv(os.path.join(self.cwd, 'data/raw/loan_data.csv'))
        print()
        print('-'*34)
        print('----- Load Data successfully -----')
        print('-'*34)
        print()
        print('Data shape : ', data.shape)

        return data
    
    def data_split_in_out(self):  # Fix return type annotation
        """
        This function do following task
        1. Encoding the categorical column
        2. Feature scaling 
        3. Splitting the data into train and test
        4. Some more transformation 

        params
        -(None)

        return
        - tuple (X, y)

        """
        data = self.load_data()

        # split data into training and testing data
        X = data.drop('loan_status', axis=1)    # Input feature
        y = data['loan_status']  
        
        return X, y   
    
    @staticmethod
    def column_transformer():
        # Numerical features
        num_features = [
            'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
        ]

        # Categorical features
        cat_features = [
            'person_gender', 'person_education', 'person_home_ownership',
            'loan_intent', 'previous_loan_defaults_on_file'
        ]

        # Create preprocessing pipelines
        # For numerical features: standardization
        num_transformer = StandardScaler()
        
        # For categorical features: use OrdinalEncoder or OneHotEncoder
        # OrdinalEncoder is better for features with ordinal relationship
        # OneHotEncoder is better for nominal features
        cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        # Alternative: Use OneHotEncoder for better performance with tree-based models
        # cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)
            ],
            remainder='passthrough'  # Keep any other columns as-is
        )

        return preprocessor, num_features, cat_features
    
    def process_data(self):
        # Fix the method call - use self.load_data() instead of static call
        X, y = self.data_split_in_out()  # This already calls load_data internally

        preprocessor, num_features, cat_features = Preprocess.column_transformer()

        # Fit and transform the data
        X_transformed = preprocessor.fit_transform(X)
        
        # Create version directory and save the fitted preprocessor
        os.makedirs(self.version_folder, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(self.version_folder, "preprocessor.pkl"))
        
        # Encode target variable separately using LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Save the fitted label encoder
        joblib.dump(label_encoder, os.path.join(self.version_folder, "label_encoder.pkl"))
        
        print('-'*40)
        print('----- Preprocessing Complete -----')
        print('-'*40)
        print(f'Version folder: {self.version_folder}')
        print(f'Original X shape: {X.shape}')
        print(f'Transformed X shape: {X_transformed.shape}')
        print(f'Number of numerical features: {len(num_features)}')
        print(f'Number of categorical features: {len(cat_features)}')
        print(f'Target classes: {label_encoder.classes_}')
        print(f'Target distribution: {pd.Series(y_encoded).value_counts().to_dict()}')
        
        # Store the preprocessor and label encoder for future use
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder

        # Split the data into the train and testing set
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42)

        print('-'*40)
        print('----- Split Data into Train and Test Set -----')
        print('-'*40)
        print(f'Train set shape: {X_train.shape}')
        print(f'Test set shape: {X_test.shape}')

        return X_train, X_test, y_train, y_test
    
    def get_version_folder(self):
        """
        Get the current version folder path
        
        return:
        - str: current version folder path
        """
        return self.version_folder
    
    def get_model_save_paths(self):
        """
        Get standard file paths for saving models and preprocessors
        
        return:
        - dict: dictionary with common file paths
        """
        return {
            'preprocessor': os.path.join(self.version_folder, 'preprocessor.pkl'),
            'label_encoder': os.path.join(self.version_folder, 'label_encoder.pkl'),
            'model': os.path.join(self.version_folder, 'model.pkl'),
            'metrics': os.path.join(self.version_folder, 'metrics.json'),
            'config': os.path.join(self.version_folder, 'config.json')
        }