"""
Author  : Rupesh Garsondiya
github  : @Rupeshgarsondiya
Organization : L.J University
"""

from gpu_config.check import GPU_Config
from config.config import Config
from nn_arch.architecture1 import ANN_arch1

if __name__ == "__main__":
    g = GPU_Config()
    g.check_gpu_configration()
    c = Config()
    c.get_config()
    ann = ANN_arch1()
    ann.ANN_arch()  # This will print the architecture of the ANN model
else :
    pass

    