"""
Author  : Rupesh Garsondiya
github  : @Rupeshgarsondiya
Organization : L.J University
"""

from gpu_config.check import GPU_Config
from data.processed import ProcessedData

class Train :

    def __init__(self):
        pass

    def model_train(self) -> None:
        print("Model is being trained")
        print()
        print("Model is trained")
        pass

if __name__ == "__main__":
    g = GPU_Config
    g.check_gpu_configration()

    p = ProcessedData()
    p.load_data()
    train = Train()
    train.model_train()  # calling the method