import torch
import pandas as pd

from torch import nn

from lightning import LightningModule
from regression.StandardScaledDataset import StandardScaledDataset

class RegressionTunableModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        
        last_out_size = config["layer1_output"]
        self.encoder = nn.Sequential(
            nn.Linear(config["input_shape"], last_out_size), 
            nn.LeakyReLU(),
        )

        dropout = config["layer1_dropout"]
        self.encoder.append(nn.Dropout(dropout))

        for layer_index in range(2, int(config["layer_count"])+1):
            next_out_size = config[f"layer{layer_index}_output"]
            self.encoder.append(nn.Linear(last_out_size, next_out_size))
            self.encoder.append(nn.LeakyReLU())
            dropout = config[f"layer{layer_index}_dropout"]
            self.encoder.append(nn.Dropout(dropout))
            last_out_size = next_out_size

        self.encoder.append(nn.Linear(last_out_size, 1))

    def forward(self, x):
        return self.encoder(x)

if __name__ == "__main__": 
    full_dataset = StandardScaledDataset()

    restored_model = RegressionTunableModel.load_from_checkpoint("checkpoint/checkpoint").eval()

    target = pd.DataFrame([0, 0, 0, 0, 30.801135, 0, 1.863269, 0, 0, 0, 0, 0, 0, 0, 0], index=full_dataset.data_columns).transpose()
    full_dataset.scale_data_columns(target)
    target = torch.FloatTensor(target.values).cuda()

    prediction = restored_model(target)

    rescaled_prediction = full_dataset.rescale_target(prediction)
    print(rescaled_prediction)