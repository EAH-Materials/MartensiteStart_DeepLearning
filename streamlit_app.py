import streamlit as st
import torch
import os
import pandas as pd
import sklearn

from matplotlib import pyplot as plt

from src.DeployModel import DeployModel
from src.utilities_steel import Ms_Ingber
from src.MS_Pycalphad import ms_Calphad
from src.RangeCompute import range_study
from src.lof_ms import LofMs


@st.cache_resource
def load_model():
    return DeployModel.load_from_checkpoint(
        "src/checkpoint/checkpoint", map_location=torch.device("cpu")
    )

@st.cache_resource
def load_data():
    return pd.DataFrame(pd.read_csv(os.path.join("data", "MsDatabase_2022.csv")))

#@st.cache_resource
def load_lof():
    return LofMs(df_data=load_data()) 


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

def plot_lof_space(lof_space,sample_value):
    st.subheader("Data Sample representation:")
    
    fig, ax = plt.subplots(1, 2,figsize=(12,3))
    
    ax[0].boxplot(lof_space,
                    vert=False,
                    #showfliers=False,
                    flierprops=dict(markerfacecolor='black', marker='x'),
                    whis=1.5,
                    patch_artist = True,
                    boxprops = dict(facecolor = "lightblue")
        )
    space = (0.9,1.1)
    ax[0].set_ylim(space)
    ax[0].plot([sample_value,sample_value],[space[0],space[1]],'--',color='r',linewidth=0.8)
    ax[0].axes.get_yaxis().set_visible(False)
    ax[0].grid()
    ax[0].set_xlabel('LOF')

    y, _, _ = ax[1].hist(lof_space,color = "black",bins=150, histtype='step')
    y, _, _ = ax[1].hist(lof_space,color = "lightblue",bins=150)
    ax[1].plot([sample_value,sample_value],[0,max(y)],'--',color='r',linewidth=0.8)
    ax[1].grid()
    ax[1].set_xlabel('LOF')
    ax[1].set_ylabel('# of samples')

    st.pyplot(fig,dpi=300)
    

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
    cols_NN, col_EM, col_TD, col_LO = st.columns(4)
    with cols_NN:
        __print_func__(predictions[0], "NN", "Neural Network:")
    with col_EM:
        __print_func__(predictions[1], "EM", "Empirical Model:")
    with col_TD:
        __print_func__(predictions[2], "TD", "Thermodynamic Model:")
    with col_LO:
        #__print_func__(predictions[3], "LO", "LOF:")
        delta = 0
        if f"previous_value_LOF" in st.session_state:
            delta = predictions[3] - st.session_state[f"previous_value_LOF"]
        st.write("**" + "Local Outlier Factor:" + "**")
        st.metric(
            f"LOF",
            f"{predictions[3]:5.2f}",
            delta=f"{delta:5.2f} ",
            label_visibility="hidden",
            delta_color="inverse",
        )
        st.write("compared to previous calculation")
        st.session_state[f"previous_value_LOF"] = predictions[3]

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
    data  = load_data()
    lof   = load_lof()
    
    
    with comp_tab:
        slider_or_numbers = st.toggle("Input with sliders or numeric inputs.",disabled=False)
        if slider_or_numbers:
            composition_vec, composition_dict = get_inputs(st.slider)
        else:
            composition_vec, composition_dict = get_inputs(st.number_input)
        
        composition_vec_trans = composition_vec.copy()
        Ms_NN = model.inference_vector(composition_vec_trans)
        Ms_EM = Ms_Ingber(**composition_dict)
        Ms_TD = ms_Calphad(**composition_dict, T_guess=Ms_EM)
        Lof_S = lof.score_samples(composition_vec)[0]
        
        st.write('\n')
        print_result([Ms_NN, Ms_EM, Ms_TD, Lof_S])
        
        st.write('\n')
        data_sample_representation = st.toggle("Data Sample representation compared to Dataset.",value=True)
        if data_sample_representation:
            plot_lof_space(lof_space = lof.dataset_lof(),sample_value=lof.score_samples(composition_vec))
        
        
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

    st.subheader("Acknowledgements:")
    st.write("The SteelDesAIn project is funded by the Carl Zeiss Foundation.")
    st.markdown("Contains information from TODO: DATABASE NAME, which is made available here under the <a target='_blank' href='https://opendatacommons.org/licenses/odbl/1-0/'>Open Database License (ODbL)</a>.", unsafe_allow_html=True)
    st.subheader("Details on the used Deep Learning Model can be found in Paper: TODO")
    st.write("Impressum: Work in Progress")
