"""
Name         : Rupesh Garsondiya
github       : @Rupeshgarsondiya
Organization : L.J University
"""

import os
import pandas as pd


class Preprocess:
    
    def __init__(self):
        self.cwd = os.getcwd()

    def load_data(self) :

        path = self.cwd+"data"
        data = pd.read_csv(os.path.join(path, 'loan_data.csv'))

        print('-'*30)
        print('----- Load Data successfully -----')
        print('-'*30)
        print()
        print()
        print('Data shape : ',data.shape)
        
        return data