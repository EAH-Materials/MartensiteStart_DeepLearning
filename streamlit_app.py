import streamlit as st
import torch

from src.DeployModel import DeployModel
    
@st.cache_resource
def load_model():
    return DeployModel.load_from_checkpoint("src/checkpoint/checkpoint", map_location=torch.device('cpu'))

if __name__ == "__main__": 
    st.image("imgs/eah_logo.jpg", width=400)
    st.title("Predicting the Martensite Start Temperature for Steel Alloys")
    st.write("Disclaimer: The results from this tool are estimates based on data consisting of a set of experimental measurements. All results are provided for informational purposes only, in furtherance of the developers' educational mission, to complement the knowledge of materials scientists and engineers, and assist them in their search for new materials with desired properties. The developers may not be held responsible for any decisions based on this tool.")

    st.write("Input in wt %. Upper limit represents the upper limits within the training data.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        c = st.slider('Carbon (C)', 0., 2.25, 0.)
        mn = st.slider('Manganese (Mn)', 0., 10.24, 0.)
        si = st.slider('Silicon (Si)', 0., 3.8, 0.)
        cr = st.slider('Chromium (Cr)', 0., 17.98, 0.)
        ni = st.slider('Nickel (Ni)', 0., 31.54, 0.)

    with col2:
        mo = st.slider('Molybdenum (Mo)', 0., 8.0, 0.)
        v = st.slider('Vanadium (V)', 0., 5.05, 0.)
        co = st.slider('Cobalt (Co)', 0., 16.08, 0.)
        al = st.slider('Aluminium (Al)', 0., 3.01, 0.)
        w = st.slider('Tungsten (W)', 0., 19.2, 0.)

    with col3:
        cu = st.slider('Copper (Cu)', 0., 3.04, 0.)
        nb = st.slider('Niobium (Nb)', 0., 1.98, 0.)
        ti = st.slider('Titanium (Ti)', 0., 2.52, 0.)
        b = st.slider('Boron (B)', 0., 0.004, 0., 0.001, format="%5.3f")
        n = st.slider('Nitrogen (N)', 0., 2.65, 0.)

    model = load_model()
    prediction = model.inference_vector([c, mn, si, cr, ni, mo, v, co, al, w, cu, nb, ti, b, n])
    
    delta = 0
    if "previous_value" in st.session_state:
        delta = prediction - st.session_state.previous_value

    st.subheader("Martensite Start Temperature:")

    _, col4, _ = st.columns(3)
    with col4:
        st.metric("MsT", f"{prediction:5.2f} K", delta=f"{delta:5.2f} K", label_visibility="hidden")
        st.write("compared to previous calculation")

    st.session_state.previous_value = prediction

    st.subheader("Acknowledgements: ")
    st.subheader("Details on the used Deep Learning Model can be found in Paper: TODO")
    st.write("Impressum: Work in Progress")