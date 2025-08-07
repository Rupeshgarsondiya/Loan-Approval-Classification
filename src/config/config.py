"""
Name         : Rupesh Garsondiya
github       : @Rupeshgarsondiya
Organization : L.J University
"""

import os

class Config:
    def __init__(self):

        # Path of current working directory and raw and processed data
        try:
            self.cwd = os.getcwd()
            self.RAW_DATA_PATH = os.path.join(self.cwd,'data/raw')
            self.PROCESSED_DATA_PATH = os.path.join(self.cwd,'data/processed')

        except Exception as e:
            print('Exception raise : ',e)
        
             # General

        self.RANDOM_STATE = 40

        # Logistic Regression
        self.PENALTY = 'l2'  # Options: 'l1', 'l2', 'elasticnet', 'none'
        self.C = 1.0         # Options: 0.1, 1.0, 10.0
        self.MAX_ITER = 100 # Options: 100, 500, 1000

        # Decision Tree
        self.CRITERION = 'gini'           # Options: 'gini', 'entropy', 'log_loss'
        self.MAX_DEPTH = 10            # Options: None, 5, 10, 20
        self.MIN_SAMPLES_SPLIT = 5       # Options: 2, 5, 10
        self.MIN_SAMPLES_LEAF = 1         # Options: 1, 2, 4

        # Random Forest
        self.N_ESTIMATORS = 100           # Options: 50, 100, 200
        self.MAX_FEATURES = 'auto'        # Options: 'auto', 'sqrt', 'log2'
        self.MAX_DEPTH_RF = 10          # Options: None, 10, 20
        self.BOOTSTARP = True             # Options: True, False
        self.N_JOBS = -1                  # Options: -1 (all cores), 1, 2

        # K-Nearest Neighbors
        self.N_NEIGHBORS = 5              # Options: 3, 5, 7
        self.ALGORITHM = 'brute'           # Options: 'auto', 'ball_tree', 'kd_tree', 'brute'
        self.P = 2                        # Options: 1 (Manhattan), 2 (Euclidean)
        self.WEIGHTS = 'distance'          # Options: 'uniform', 'distance'
        self.METRIC = 'minkowski'         # Options: 'minkowski', 'euclidean', 'manhattan'

        # Neural Network
        self.BATCH_SIZE = 64              # Options: 16, 32, 64
        self.LEARNING_RATE = 0.001        # Options: 0.0001, 0.001, 0.01
        self.EPOCHS = 50                  # Options: 20, 50, 100
        self.DROPOUT = 0.3                # Options: 0.2, 0.3, 0.5


    def get_config(self) -> None:
        """
        This function get the configration of a project

        params

        -(None)

        return
        -(None)
        """
        print('-'*34)
        print(f'configration')
        print('-'*34)
        print(f"""
        Current working directory: {self.cwd}
        Raw data path: {self.RAW_DATA_PATH}
        Processed data path: {self.PROCESSED_DATA_PATH}
        Batch Size: {self.BATCH_SIZE}
        Learning Rate: {self.LEARNING_RATE}
        Epochs: {self.EPOCHS}
        Dropout ratio: {self.DROPOUT}
        """)


        
