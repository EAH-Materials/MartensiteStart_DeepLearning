import torch
from torch.utils.data import Dataset

from pandas import DataFrame

import numpy as np

# %%
class StandardScaledDataset(Dataset):
    def __init__(self):
        self.data_columns = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Co", "Al", "W", "Cu", "Nb", "Ti", "B", "N"]
        self.target_columns = ["Ms"]

        self.data_mean, self.data_std = self.calculate_scale_metrics()

        self.target_mean = 587.6420457809462
        self.target_std = 107.61116120718675

    def rescale_target(self, value):
        return value*self.target_std + self.target_mean
    
    def rescale(self, values):
        for i, column in enumerate(self.data_columns):
            value = values[:, i]
            values[:, i] = np.round(value*self.data_std[column] + self.data_mean[column], 4)
        return values
    
    def scale_data_columns(self, df):
        for column_name in df.columns:
            df[column_name] = self.scale_data_column(df, column_name)

    def scale_data_column(self, df, column):
        return self._scale(df[column], self.data_mean[column], self.data_std[column])
    
    def _scale(self, value, mean, std):
        return (value - mean) / std

    def calculate_scale_metrics(self):
        mean, std = {}, {}

        mean["C"] = 0.35294931950745295
        mean["Mn"] = 0.7624457096565134
        mean["Si"] = 0.332420194426442
        mean["Cr"] = 1.1030602851587816
        mean["Ni"] = 2.6435700129617627
        mean["Mo"] = 0.29161202203499675
        mean["V"] = 0.10516671419313027
        mean["Co"] = 0.18455424497731693
        mean["Al"] = 0.020678094620868438
        mean["W"] = 0.37233430330524947
        mean["Cu"] = 0.044665528191834086
        mean["Nb"] = 0.00668602721970188
        mean["Ti"] = 0.009008094620868438
        mean["B"] = 9.397278029812057e-06
        mean["N"] = 0.014538755670771227

        std["C"] = 0.2941571774906254
        std["Mn"] = 0.6501711392501798
        std["Si"] = 0.4029904606069103
        std["Cr"] = 2.211743845702123
        std["Ni"] = 6.342158791190849
        std["Mo"] = 0.6418696825408096
        std["V"] = 0.3884091859639032
        std["Co"] = 1.355922392821505
        std["Al"] = 0.14680902309859412
        std["W"] = 2.0992142449601774
        std["Cu"] = 0.17014241185441442
        std["Nb"] = 0.0931867931167249
        std["Ti"] = 0.11709327779176554
        std["B"] = 0.00015643096932446688
        std["N"] = 0.1416612051493458

        return mean, std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx=0):
        x = self.data[idx]
        y = self.target[idx]
        return x, y