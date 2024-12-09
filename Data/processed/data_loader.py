'''
Name    : Rupesh Garsondiya
github  : @Rupeshgarsondiya
Organization : L.J University
'''

import os
import pandas as pd

def load_loan_data():
    """Load loan_data.csv from the raw folder."""
    # Define the path relative to this script's location
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level
    raw_data_path = os.path.join(base_path, "raw", "loan_data.csv")
    
    # Check if the file exists
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"The file {raw_data_path} does not exist.")
    
    # Load the CSV file
    data = pd.read_csv(raw_data_path)
    print(f"Loaded data from {raw_data_path}")
    return data

if __name__ == "__main__":
    # Load the data and display its head
    loan_data = load_loan_data()
    print(loan_data.head())
