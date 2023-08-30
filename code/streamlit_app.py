import streamlit as st
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
    st.title("Predicting the Martensite Start Temperature for Steel Alloys")
    st.subheader("Details on the used Deep Learning Model can be found in Paper: TODO")
    full_dataset = StandardScaledDataset()

    previous_value = 0

    restored_model = RegressionTunableModel.load_from_checkpoint("checkpoint/checkpoint").eval()

    col1, col2, col3 = st.columns(3)
    
    with col1:
        c = st.slider('C', 0., 2.25, 0.)
        mn = st.slider('Mn', 0., 10.24, 0.)
        si = st.slider('Si', 0., 3.8, 0.)
        cr = st.slider('Cr', 0., 17.98, 0.)
        ni = st.slider('Ni', 0., 31.54, 0.)

    with col2:
        mo = st.slider('Mo', 0., 8.0, 0.)
        v = st.slider('V', 0., 5.05, 0.)
        co = st.slider('Co', 0., 16.08, 0.)
        al = st.slider('Al', 0., 3.01, 0.)
        w = st.slider('W', 0., 19.2, 0.)

    with col3:
        cu = st.slider('Cu', 0., 3.04, 0.)
        nb = st.slider('Nb', 0., 1.98, 0.)
        ti = st.slider('Ti', 0., 2.52, 0.)
        b = st.slider('B', 0., 0.004, 0., 0.001)
        n = st.slider('N', 0., 2.65, 0.)

    target = pd.DataFrame([c, mn, si, cr, ni, mo, v, co, al, w, cu, nb, ti, b, n], index=full_dataset.data_columns).transpose()
    full_dataset.scale_data_columns(target)
    target = torch.FloatTensor(target.values).cuda()

    prediction = restored_model(target)

    rescaled_prediction = full_dataset.rescale_target(prediction).item()
    rescaled_prediction = f"{rescaled_prediction:10.2f} K"

    st.subheader("Martensite Start Temperature:")
    st.metric("MsT", rescaled_prediction, delta=rescaled_prediction, label_visibility="hidden")
    previous_value = rescaled_prediction

    st.write("Disclaimer...")