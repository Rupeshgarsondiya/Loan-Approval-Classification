"""
Name         : Rupesh Garsondiya
github       : @Rupeshgarsondiya
Organization : L.J University
"""

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras._tf_keras.keras.layers import Dense,Dropout,BatchNormalization
from scripts.preprocess_data import Preprocess


class ANN_arch1:
    
    def __init__(self):
        pass

    def ANN_arch(self) -> object :
        """
        This function is sepcial for the neural network architecture

        params

        -(None)

        Return
        -(Object of sequential model)
        """
        p = Preprocess()
        data = p.load_data()
        

        # design the neural network
        model = Sequential()

        model.add(Dense(64,activation='relu',input_dim = data.shape[0]))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam',loss='mean_squared_error')
        model.summary()
        return model