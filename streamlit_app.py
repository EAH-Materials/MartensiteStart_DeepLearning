import streamlit as st
import torch

from src.DeployModel import DeployModel
    
@st.cache_resource
def load_model():
    return DeployModel.load_from_checkpoint("src/checkpoint/checkpoint", map_location=torch.device('cpu'))

def define_input(element):
    col1, col2, col3 = st.columns(3)
    with col1:
        c = element('Carbon (C)', 0., 2.25, 0.)
        mn = element('Manganese (Mn)', 0., 10.24, 0.)
        si = element('Silicon (Si)', 0., 3.8, 0.)
        cr = element('Chromium (Cr)', 0., 17.98, 0.)
        ni = element('Nickel (Ni)', 0., 31.54, 0.)

    with col2:
        mo = element('Molybdenum (Mo)', 0., 8.0, 0.)
        v = element('Vanadium (V)', 0., 5.05, 0.)
        co = element('Cobalt (Co)', 0., 16.08, 0.)
        al = element('Aluminium (Al)', 0., 3.01, 0.)
        w = element('Tungsten (W)', 0., 19.2, 0.)

    with col3:
        cu = element('Copper (Cu)', 0., 3.04, 0.)
        nb = element('Niobium (Nb)', 0., 1.98, 0.)
        ti = element('Titanium (Ti)', 0., 2.52, 0.)
        b = element('Boron (B)', 0., 0.004, 0., 0.001, format="%5.3f")
        n = element('Nitrogen (N)', 0., 2.65, 0.)
    prediction = model.inference_vector([c, mn, si, cr, ni, mo, v, co, al, w, cu, nb, ti, b, n])
    return prediction

def define_sliders():
    prediction = define_input(st.slider)
    define_result(prediction, "slider")

def define_numeric():
    prediction = define_input(st.number_input)
    define_result(prediction, "numbers")

def define_result(prediction, key):
    st.subheader("Martensite Start Temperature:")
    
    if prediction <= 0:
        st.write("The model prediction is very inaccuarate for this alloy (prediction resulted in value below 0 K). Please try a different alloy.")
    else:
        
        delta = 0
        if f"previous_value{key}" in st.session_state:
            delta = prediction - st.session_state[f"previous_value{key}"]

        col4, _ = st.columns(2)
        with col4:
            st.metric("MsT", f"{prediction:5.2f} K / {(prediction-273.15):5.2f} °C", delta=f"{delta:5.2f} K/°C", label_visibility="hidden")
            st.write("compared to previous calculation")

        st.session_state[f"previous_value{key}"] = prediction

if __name__ == "__main__": 
    st.image("imgs/eah_logo.jpg", width=300)
    st.title("Predicting the Martensite Start Temperature for Steels")
    st.write("Disclaimer: The developers may not be held responsible for any decisions based on this tool. The model and results are provided for informational and educational purpose only, to assist in the search for new materials with desired properties. The provided tool estimates the results based on publicy available data from a set of experimental measurements.")

    st.write("Input in wt %. Upper limit represents the upper limits within the training data.")
    
    slider, numbers, others, data = st.tabs(["Slider", "Numbers", "Other models", "Data"])

    model = load_model()

    with slider:
        define_sliders()

    with numbers:
        define_numeric()

    st.subheader("Acknowledgements: ")
    st.subheader("Details on the used Deep Learning Model can be found in Paper: TODO")
    st.write("Impressum: Work in Progress")