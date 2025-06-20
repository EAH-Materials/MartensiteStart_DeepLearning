import streamlit as st
import torch
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.DeployModel import DeployModel
from src.utilities_steel import Ms_Ingber
from src.MS_Pycalphad import ms_Calphad
from src.RangeCompute import range_study_1D, range_study_2D
from src.lof_ms import LofMs


@st.cache_resource
def load_model():
    return DeployModel.load_from_checkpoint(
        "src/checkpoint/checkpoint", map_location=torch.device("cpu")
    )


@st.cache_resource
def load_data():
    return pd.DataFrame(pd.read_csv(os.path.join("data", "MsDatabase_2022.csv")))

# @st.cache_resource
def load_lof(df_data=None):
    if df_data is None:
        df_data = load_data()
    return LofMs(df_data=df_data)


disclaimer = "Limitation of liability for website content: The developers may not be held responsible for any decisions based on this tool. The model and results are provided for informational and educational purpose only, to assist in the search for new materials with desired properties. The provided tool estimates the results based on publicy available data from a set of experimental measurements. The contents were created with the highest standard of applicable care and to the best of the developers knowledge. Nevertheless, the accuracy of the content cannot be guaranteed."


def get_inputs(element):
    col1, col2, col3 = st.columns(3)
    with col1:
        c = element("Carbon (C)", 0.0, 2.25, 0.0, )
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


def plot_lof_space(lof_space, sample_value):

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Box Plot", "Histogram"))
    fig.add_trace(
        go.Box(x=lof_space, name="", showlegend=False, marker_color="#009898"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=lof_space,
            nbinsx=150,
            name="",
            showlegend=False,
            marker_color="#009898",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=sample_value.squeeze(), line_dash="dash", line_color="red", row=1, col="all"
    )
    if max(lof_space) >= sample_value.squeeze():
        upper_xlim = max(lof_space)
    else:
        upper_xlim = sample_value.squeeze()
    
    fig.update_layout(
        xaxis=dict(showgrid=True, title_text="LOF",range=[0,upper_xlim+0.2]),
        xaxis2=dict(showgrid=True, title_text="LOF",range=[0,upper_xlim+0.2]),
        yaxis=dict(showgrid=True),
        yaxis2=dict(showgrid=True, title_text="# of samples"),
    )
    st.plotly_chart(fig, use_container_width=True)


def print_result(predictions):
    def __print_func__(value, suffix, head):
        delta = 0
        if f"previous_value_{suffix}" in st.session_state:
            delta = value - st.session_state[f"previous_value_{suffix}"]
        st.markdown("#### **" + head + "**")
        if suffix == "LO":
            st.metric(
                f"LOF",
                f"{predictions[3]:5.2f}",
                delta=f"{delta:5.2f}",
                label_visibility="hidden",
                delta_color="inverse",
            )
            st.write("compared to previous calculation")
            st.session_state[f"previous_value_{suffix}"] = value
        else:
            if value >= 0.0:
                st.metric(
                    f"MsT_{suffix}",
                    f"{value:5.2f} K | {(value-273.15):5.2f} °C",
                    delta=f"{delta:5.2f} K|°C",
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
        __print_func__(predictions[0], "NN", "Artificial Neural Network:")
        with st.expander("ⓘ Expand for Details"):
            st.markdown(
                "[[Paper](https://doi.org/10.1002/srin.202400210) | [Code](https://github.com/EAH-Materials/MartensiteStart_DeepLearning)] This artificial neural network had been trained on 1,500 records of experimental data, $M_s$ ranging from -123 °C to 790 °C. Be aware, the model does not know about physical limitations and may thus predict a temperature lower than 0 K or differ by a lot from the other models for alloys that are not well represented in our training data."
            )
    with col_EM:
        __print_func__(predictions[1], "EM", "Empirical Model:")
        with st.expander("ⓘ Expand for Details"):
            st.markdown(
                "This [empirical model](https://onlinelibrary.wiley.com/doi/full/10.1002/srin.202100576) estimates the martensite start temperature as a function of chemical composition. It is developed and optimized for high-carbon steels with a $M_s$ range between 0 and 50 °C. It is based on the mean value of several empirical models from the literature. Click ?-symbol on right for the exact formular.",
                help="$M_{\\text{s}}=\\frac{1}{8}\\left\\{4241.9-2322.27x_{\\text{C}}-284x_{\\text{Mn}}- \n 54.4x_{\\text{Si}}-166.4x_{\\text{Cr}}-137.4x_{\\text{Ni}}-83.5x_{\\text{Mo}}-30x_{\\text{Al}}+38.58x_{\\text{Co}}-600 \\lbrack 1-exp(-0.96x_{\\text{C}}) \\rbrack \\right\\}$",
            )
    with col_TD:
        __print_func__(predictions[2], "TD", "Thermodynamic Model:")
        with st.expander("ⓘ Expand for Details"):
            st.markdown(
                "This model uses [pyCALPHAD](https://pycalphad.org/) to compute Gibbs energies for Austenite and Martensite over a large temperature range and the [Ghosh-Olson model](https://link.springer.com/article/10.1361/105497101770338653) to compute the driving force for martensitic transformation $Δ_G$."
            )
            st.image("imgs/Gibbs_vs_T.png")
    with col_LO:
        __print_func__(predictions[3], "LO", "Data Sample representation:")
        with st.expander("ⓘ Expand for Details"):
            st.markdown(
                "Local Outlier Factor [LOF](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf) is used to represent the Input Data Sample compared to the Dataset. \n The local outlier factor is based on a concept of a local density, where locality is given by k nearest neighbors, whose distance is used to estimate the density. By comparing the local density of an object to the local densities of its neighbors, one can identify regions of similar density, and points that have a substantially lower density than their neighbors. \n LOF is defined between 0 and ∞, while 1 means that the local density of an object is equal to the local densities of its neighbors. Data Samples with a low LOF value are well represented by the Dataset, while those with a high LOF value are not. "
            )

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
                    height: 80vh !important;
                }
                .plot-container {
                    height: 81vh !important;
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
        st.image("imgs/eah_logo.jpg", width=150)
    with col3:
        st.image("imgs/Carl-Zeiss-Stiftung_Logo.png", width=75)
    st.title("Predicting the Martensite Start Temperature for Steels")

    st.write(
        "All composition values in wt %. Upper limits represent the maximum values in the training data."
    )
    comp_tab, range_tab, data_tab, info_tab = st.tabs(
        ["Calculator", "Range Calculator", "Data", "Publication information"]
    )

    model = load_model()
    data = load_data()
    lof = load_lof(data)

    with comp_tab:
        slider_or_numbers = st.toggle(
            "Input with sliders or numeric inputs.", disabled=False
        )
        if slider_or_numbers:
            composition_vec, composition_dict = get_inputs(st.slider)
        else:
            composition_vec, composition_dict = get_inputs(st.number_input)

        composition_vec_trans = composition_vec.copy()
        Ms_NN = model.inference_vector(composition_vec_trans)
        Ms_EM = Ms_Ingber(**composition_dict)
        Ms_TD = ms_Calphad(**composition_dict, T_guess=Ms_EM)
        Lof_S = lof.score_samples(composition_vec)[0]

        st.write("\n")
        print_result([Ms_NN, Ms_EM, Ms_TD, Lof_S])

        st.write("\n")
        st.subheader("Data Sample representation:")
        data_sample_representation = st.toggle(
            "Data Sample representation compared to Dataset.", value=False
        )
        if data_sample_representation:
            st.write("Visualize the LOF Value of the Data Sample compared the LOF Values of all Datapoints in the Dataset.")
            plot_lof_space(
                lof_space=lof.dataset_lof(),
                sample_value=lof.score_samples(composition_vec),
            )

    with range_tab:
        fig = None
        col1, col2 = st.columns(2)
        with col1:
            if st.toggle('Use reference composition', False, help="The reference composition can be set in the 'Calculator'-tab. All computations are then based on this given composition, varying the elements given in this tab. If this option is deactivated all values besides the given ranges are assumed to be zero."):
                composition_dict_rng = composition_dict
            else:
                composition_dict_rng = None
        with col2:
            dim2d = st.toggle("1D/2D Range", False)
        if composition_dict_rng is not None:
            st.dataframe(data=pd.DataFrame(composition_dict_rng,index=[0]), height=32, use_container_width=True,hide_index=True)
        if dim2d:
            col3, col4 = st.columns(2)
            with col3:
                e1 = st.selectbox("Element 1", model.order, 0)
                e1_lb = st.number_input("Lower Bound 1", 0.0, None, 0.0)
                e1_ub = st.number_input("Upper Bound 1", 0.0, None, 1.0)
            with col4:
                e2 = st.selectbox("Element 2", model.order, 1)
                e2_lb = st.number_input("Lower Bound 2", 0.0, None, 0.0)
                e2_ub = st.number_input("Upper Bound 2", 0.0, None, 1.0)
        else:
            e1 = st.selectbox("Element", model.order, 0)
            e1_lb = st.number_input("Lower Bound", 0.0, None, 0.0)
            e1_ub = st.number_input("Upper Bound", 0.0, None, 1.0)

        col5, col6, col7, col8, col9 = st.columns(5)
        models = []
        with col5:
            comp = st.button("Compute")
        with col6:
            if st.checkbox("Artificial Neural Network", value=True):
                models.append("NN")
        with col7:
            if st.checkbox("Empirical Model"):
                models.append("EM")
        with col8:
            if st.checkbox("Thermodynamic Model"):
                models.append("TD")
        with col9:
            mea_val_th = st.number_input("Measured Data Threshold", 0.0, 1.0, 0.05, 0.01, help="Relative threshold for measured data display in the range study. Default is 0.05 (5% of the maximum value within measured dataset). The points in the graph will be colored according to how well the composition matches the reference composition under consideration (black: good match -> white: bad match).")

        if comp:
            fig = None
            with st.spinner("Computing"):
                if dim2d:
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
                            "ref":composition_dict_rng
                        }
                    ]
                    fig = range_study_2D(study, df=data, models=models,threshold_measured_data_display=mea_val_th)
                else:
                    study = [
                        {
                            "e1": {
                                "element": e1,
                                "min": e1_lb,
                                "max": e1_ub,
                                "sample_points": 16,
                            },
                            "ref":composition_dict_rng
                        }
                    ]
                    fig = range_study_1D(study, df=data, models=models, threshold_measured_data_display=mea_val_th)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with data_tab:
        st.dataframe(data, use_container_width=True, height=600)

    with info_tab:
        st.subheader("Availability & further Information:")
        st.markdown(
            "Details on the used Deep Learning Model can be found in our Paper [Machine Learning-Based Prediction of the Martensite Start Temperature](https://doi.org/10.1002/srin.202400210)"
        )
        st.markdown(
            "This website and the deep learning model are open-source and published under [GNU GPLv3](https://github.com/EAH-Materials/MartensiteStart_DeepLearning/blob/main/LICENSE) on GitHub: [https://github.com/EAH-Materials/MartensiteStart_DeepLearning](https://github.com/EAH-Materials/MartensiteStart_DeepLearning).",
        )
        st.subheader("Acknowledgements:")
        st.markdown("The SteelDesAIn project is funded by the [Carl Zeiss Foundation](https://www.carl-zeiss-stiftung.de/en/).")
        st.markdown(
            "Contains information from the thermodynamic database 'mc_fe_v2.059.tdb', which is made available here under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).",
        )

    st.divider()
    st.markdown(
        "[Imprint (Impressum)](https://www.eah-jena.de/impressum) (Forwards to the website of the University of Applied Sciences Jena in a new tab)"
    )
    st.markdown(disclaimer)
