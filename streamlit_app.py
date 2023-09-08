import streamlit as st
import torch
import os
import pandas as pd

from src.DeployModel import DeployModel
from src.utilities_steel import Ms_Ingber
from src.MS_Pycalphad import ms_Calphad
from src.RangeCompute import range_study


@st.cache_resource
def load_model():
    return DeployModel.load_from_checkpoint(
        "src/checkpoint/checkpoint", map_location=torch.device("cpu")
    )


@st.cache_resource
def load_data():
    return pd.DataFrame(pd.read_csv(os.path.join("data", "MsDatabase_2022.csv")))


def get_inputs(element):
    col1, col2, col3 = st.columns(3)
    with col1:
        c = element("Carbon (C)", 0.0, 2.25, 0.0)
        mn = element("Manganese (Mn)", 0.0, 10.24, 0.0)
        si = element("Silicon (Si)", 0.0, 3.8, 0.0)
        cr = element("Chromium (Cr)", 0.0, 17.98, 0.0)
        ni = element("Nickel (Ni)", 0.0, 31.54, 0.0)

    with col2:
        mo = element("Molybdenum (Mo)", 0.0, 8.0, 0.0)
        v = element("Vanadium (V)", 0.0, 5.05, 0.0)
        co = element("Cobalt (Co)", 0.0, 16.08, 0.0)
        al = element("Aluminium (Al)", 0.0, 3.01, 0.0)
        w = element("Tungsten (W)", 0.0, 19.2, 0.0)

    with col3:
        cu = element("Copper (Cu)", 0.0, 3.04, 0.0)
        nb = element("Niobium (Nb)", 0.0, 1.98, 0.0)
        ti = element("Titanium (Ti)", 0.0, 2.52, 0.0)
        b = element("Boron (B)", 0.0, 0.004, 0.0, 0.001, format="%5.3f")
        n = element("Nitrogen (N)", 0.0, 2.65, 0.0)

    vec = [c, mn, si, cr, ni, mo, v, co, al, w, cu, nb, ti, b, n]
    dic = {
        "C": c,
        "MN": mn,
        "SI": si,
        "CR": cr,
        "NI": ni,
        "MO": mo,
        "V": v,
        "CO": co,
        "AL": al,
        "W": w,
        "CU": cu,
        "NB": nb,
        "TI": ti,
        "B": b,
        "N": n,
    }
    return vec, dic


def print_result(predictions):
    def __print_func__(value, suffix, head):
        delta = 0
        if f"previous_value_{suffix}" in st.session_state:
            delta = value - st.session_state[f"previous_value_{suffix}"]
        st.write("**" + head + "**")
        if value >= 0.0:
            st.metric(
                f"MsT_{suffix}",
                f"{value:5.2f} K / {(value-273.15):5.2f} °C",
                delta=f"{delta:5.2f} K/°C",
                label_visibility="hidden",
            )
            st.write("compared to previous calculation")
            st.session_state[f"previous_value_{suffix}"] = value
        else:
            st.write(
                "The model prediction is very inaccuarate for this alloy (prediction resulted in value below 0 K). Please try a different alloy."
            )

    st.subheader("Martensite Start Temperatures:")
    cols_NN, col_EM, col_TD = st.columns(3)
    with cols_NN:
        __print_func__(predictions[0], "NN", "Neural Network:")
    with col_EM:
        __print_func__(predictions[1], "EM", "Empirical Model:")
    with col_TD:
        __print_func__(predictions[2], "TD", "Thermodynamic Model:")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Predicting the Martensite Start Temperature for Steels",
        page_icon="imgs/Logo_RGB_farbig.png",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    # Inject some css styling
    st.markdown(
        """
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 0rem;
                    padding-right: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                }
                .stPlotlyChart {
                    height: 90vh !important;
                    border: 1px solid #000;
                }
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col3 = st.columns(2)
    with col1:
        st.image("imgs/eah_logo.jpg", width=300)
    with col3:
        st.image("imgs/Carl-Zeiss-Stiftung_Logo.png", width=150)
    st.title("Predicting the Martensite Start Temperature for Steels")
    st.write(
        "Disclaimer: The developers may not be held responsible for any decisions based on this tool. The model and results are provided for informational and educational purpose only, to assist in the search for new materials with desired properties. The provided tool estimates the results based on publicy available data from a set of experimental measurements."
    )

    st.write(
        "Input in wt %. Upper limit represents the upper limits within the training data."
    )
    comp_tab, range_tab, data_tab = st.tabs(["Calculator", "Range Calculator", "Data"])

    model = load_model()
    data = load_data()

    with comp_tab:
        slider_or_numbers = st.toggle("Input with sliders or numeric inputs.")
        if slider_or_numbers:
            composition_vec, composition_dict = get_inputs(st.slider)
        else:
            composition_vec, composition_dict = get_inputs(st.number_input)

        Ms_NN = model.inference_vector(composition_vec)
        Ms_EM = Ms_Ingber(**composition_dict)
        Ms_TD = ms_Calphad(**composition_dict, T_guess=Ms_EM)

        print_result([Ms_NN, Ms_EM, Ms_TD])

    with range_tab:
        fig = None
        col1, col2 = st.columns(2)
        with col1:
            e1 = st.selectbox("Element 1", model.order, 0)
            e1_lb = st.number_input("Lower Bound 1", 0.0, None, 0.0)
            e1_ub = st.number_input("Upper Bound 1", 0.0, None, 1.0)
        with col2:
            e2 = st.selectbox("Element 2", model.order, 1)
            e2_lb = st.number_input("Lower Bound 2", 0.0, None, 0.0)
            e2_ub = st.number_input("Upper Bound 2", 0.0, None, 1.0)
        comp = st.button("Compute")
        if comp:
            fig = None
            with st.spinner("Computing"):
                study = [
                    {
                        "e1": {
                            "element": e1,
                            "min": e1_lb,
                            "max": e1_ub,
                            "sample_points": 16,
                        },
                        "e2": {
                            "element": e2,
                            "min": e2_lb,
                            "max": e2_ub,
                            "sample_points": 16,
                        },
                    }
                ]
                fig = range_study(study,df=data)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with data_tab:
        st.dataframe(data, use_container_width=True)

    st.subheader("Acknowledgements: ")
    st.subheader("Details on the used Deep Learning Model can be found in Paper: TODO")
    st.write("Impressum: Work in Progress")
