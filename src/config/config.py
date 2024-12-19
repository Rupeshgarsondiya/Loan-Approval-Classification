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
        # Model Parameter
        self.BATCH_SIZE = 32       # set batch size for the training
        self.LEARNING_RATE = 0.01  # set learning rate for the training
        self.EPOCHS = 100          # epochs for the training
        self.DROPOUT = 0.2         # dropout ration in dropout layer

        # weight and biases config 
        self.ENTITY = 'neuralninjas' # set team/organization name for wandb account
        self.PROJECT = 'Loan Approval Classification ' # set project name for wandb account
        self.REINIT = True # set boolean value for reinitialization
        self.ANONYMOUS = 'allow' # set anonymous value type
        self.LOG_MODEL = 'all' # set log model type

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


        
