"""
Name         : Rupesh Garsondiya
github       : @Rupeshgarsondiya
Organization : L.J University
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing  import OneHotEncoder

class Preprocess:
    
    def __init__(self):
        self.cwd = os.getcwd()

    def load_data(self) :
        """
        This function work load the data from raw data folder

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
        print('Data shape : ',data.shape)

        return data
    def Preprocess(self) -> object:
        """
        This function do following task
        1. Encoding the cetegorical column
        2. Feature scalling 
        3. Splitting the data into train and test
        4. Some more transformation 

        params
        -(None)

        return
        - numpy array (train and test data)

        """
        data = self.load_data()

        # split data into training and testing data
        X = data.drop('loan_status', axis=1)    # Input feature
        y = data['loan_status']                 # Target variable

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        print()
        print('X train : ',X_train.shape)
        print('X train : ',y_train.shape)
        print('X train : ',X_test.shape)
        print('X train : ',y_test.shape)



        # 1 . Encoding cetegorical column
        ohe = OneHotEncoder()
        X_train_encoded = ohe.fit_transform(X_train[['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']])

        X_test_encoded = ohe.fit_transform(X_test[['person_gender','person_education','person_home_ownership','loan_intent','previous_loan_defaults_on_file']])

        print('X_train encoded : ',X_train_encoded.shape)
        print('X_test encoded : ',X_test_encoded.shape)



if __name__=='__main__':
    preprocess = Preprocess()
    data = preprocess.Preprocess()
    