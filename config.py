import os

class ParamConfig:
    def __init__(self):
        #Number of runs per model
        self.n_bags = 10

        #Each timestep/nion combination creates a model
        self.timesteps = [2000,3000,4000]
        self.nions = [80,100,120]

        self.ensemble_method = ['mean']

        self.train_data_path = 'input/train_features/'
        self.val_data_path = 'input/val_features/'
        self.test_data_path = 'input/test_features/'
        self.train_label_path = 'input/train_labels.csv'
        self.val_label_path = 'input/val_labels.csv'
        self.model_path = 'trained_models'
        self.log_file = 'logs/log.csv'



## initialize a param config
config = ParamConfig()