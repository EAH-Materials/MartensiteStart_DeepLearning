import numpy as np
import pandas as pd

from tqdm import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import cpu_count

from src.DeployModel import DeployModel
from src.utilities_steel import Ms_Ingber, Agrawal
from src.MS_Pycalphad import ms_Calphad as Ms_Calphad

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def import_filter_measured(
    path, range_1, range_2=None, threshold=0.03, df=None, ref=None
):
    if df is None:
        df = pd.read_csv(path)
    rest_elements = {
        "C": 0.0,
        "Mn": 0.0,
        "Si": 0.0,
        "Cr": 0.0,
        "Ni": 0.0,
        "Mo": 0.0,
        "V": 0.0,
        "Co": 0.0,
        "Al": 0.0,
        "W": 0.0,
        "Cu": 0.0,
        "Nb": 0.0,
        "Ti": 0.0,
        "B": 0.0,
        "N": 0.0,
    }
    if ref is not None:
        for key, value in ref.items():
            if key.capitalize() in rest_elements:
                rest_elements[key.capitalize()] = value

    rest_elements.pop(range_1["element"].capitalize())
    if range_2 is not None:
        rest_elements.pop(range_2["element"].capitalize())
        range_mask = (
            df[range_1["element"]].between(range_1["min"], range_1["max"])
        ) & (df[range_2["element"]].between(range_2["min"], range_2["max"]))
    else:
        range_mask = df[range_1["element"]].between(range_1["min"], range_1["max"])

    ref_mask = np.all(
        [
            (
                (df[element] >= (rest_elements[element] - threshold * df[element].max()))
                & (df[element] <= (rest_elements[element] + threshold * df[element].max()))
            )
            for element in rest_elements
        ],
        axis=0,
    )
    filtered_df = df[range_mask & ref_mask]
    diff = np.abs(filtered_df[list(rest_elements.keys())] - list(rest_elements.values())).sum(axis=1)
    min_diff = diff.min()
    max_diff = diff.max()
    normalized_diff = (diff - min_diff) / (max_diff - min_diff)
    normalized_diff = normalized_diff * (1.0 - 0.1) + 0.1
    hover_text = []
    for _, row in filtered_df.iterrows():
        text = "<b>Composition:</b><br>"
        for col_name, col_value in row.items():
            if col_name != 'Ms':
                text += f"{col_name}: {col_value}<br>"
        hover_text.append(text)

    return filtered_df, normalized_diff, hover_text


def comp_dict_vec_1D(e1, e1_val, order, ref=None):
    composition_dict = {}
    composition_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if ref is not None:
        for key, val in ref.items():
            if val > 0.0:
                composition_dict[key.capitalize()] = val
                composition_vec[order.index(key.capitalize())] = val
    composition_dict[e1["element"].capitalize()] = e1_val
    composition_vec[order.index(e1["element"].capitalize())] = e1_val
    return composition_dict, composition_vec


def comp_dict_vec_2D(e1, e2, e1_val, e2_val, order, ref=None):
    composition_dict = {}
    composition_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if ref is not None:
        for key, val in ref.items():
            if val > 0.0:
                composition_dict[key.capitalize()] = val
                composition_vec[order.index(key.capitalize())] = val
    composition_dict[e1["element"].capitalize()] = e1_val
    composition_dict[e2["element"].capitalize()] = e2_val
    composition_vec[order.index(e1["element"].capitalize())] = e1_val
    composition_vec[order.index(e2["element"].capitalize())] = e2_val
    return composition_dict, composition_vec


def remove_outlier(array, threshold=5):
    if np.nanstd(array) == 0:
        return array
    else:
        zscore = (array - np.nanmean(array)) / np.nanstd(array)
        array[np.abs(zscore) > threshold] = np.nan
        array[array < 0.0] = np.nan
        return array


shortcut_to_long = {
    "EM": "Empirical (Ingber)",
    "NN": "Neural Network",
    "TD": "Thermodynamic",
    "AG": "Agrawal et al."
}


def range_study_1D(studies, models=["NN", "EM", "TD"], df=None, Ms_ML=None, threshold_measured_data_display=0.03):
    """
    Study the dependence of Martensite start temperature (Ms) on the composition of a steel alloy.

    This function conducts a study to analyze how the Martensite start temperature (Ms) of a steel composition
    depends on the variation in the composition of a specific element (e1). It can use different computational
    models (e.g., thermodynamic, empirical, neural network) to predict Ms and compare the results with measured data.

    Args:
        studies (list of dict): A list of dictionaries, each representing a study with specific parameters.
            Each study dictionary should include:
            - "e1" (dict): Information about the composition range of the element, including:
                - "element" (str): The element's symbol (e.g., "C", "N").
                - "min" (float): The minimum composition value.
                - "max" (float): The maximum composition value.
                - "sample_points" (int): The number of sample points within the range.
            - "ref" (dict, optional): An optional reference composition dictionary.

        models (list, optional): A list of models to use for computing Ms (default: ["NN", "EM", "TD"]).
        df (DataFrame, optional): An optional DataFrame for plotting additional data processing (default: None).
        Ms_ML (object, optional): An instance of a machine learning model for Ms prediction (default: None).
        threshold_measured_data_display (float, optional): The threshold for filtering measured data for display

    Returns:
        plotly.graph_objs.Figure: A Plotly figure containing subplots displaying Ms dependence on composition
        for each study, computed using different models and compared with measured data.

    Notes:
        - The function uses parallel processing to efficiently compute Ms for a range of composition values.
        - The results are visualized in a multi-plot figure for easy comparison.

    Example:
        # Define study parameters
        studies = [
            {
                "e1": {"element": "C", "min": 0.0, "max": 1.5, "sample_points": 16},
                "ref": None,
            },
            {
                "e1": {"element": "N", "min": 0.0, "max": 1.5, "sample_points": 16},
                "ref": composition_dict,
            },
        ]

        # Compute and visualize Ms dependence
        fig = range_study_1D(studies, models=["NN", "TD"])
        fig.show()
    """
    pool = ProcessPoolExecutor(max_workers=cpu_count())
    if Ms_ML is None:
        Ms_ML = DeployModel.load_from_checkpoint(
            "src/checkpoint/checkpoint", map_location="cpu"
        )

    fig = make_subplots(rows=1, cols=len(studies))
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c", "#4a6528"]

    for sx, study in enumerate(studies):
        e1 = study["e1"]
        if "ref" not in study:
            study["ref"] = None
        e1_rng = np.linspace(e1["min"], e1["max"], e1["sample_points"])
        Ms = {model: np.zeros([e1["sample_points"]], float) for model in models}
        futures = {"futures": [], "keys": []}
        prog = tqdm(
            None,
            total=e1["sample_points"] * len(models),
            desc="Study " + str(sx + 1) + "/" + str(len(studies)),
        )
        for e1_val in e1_rng:
            composition_dict, composition_vec = comp_dict_vec_1D(
                e1, e1_val, Ms_ML.order, study["ref"]
            )
            if "TD" in models:
                futures["futures"].append(pool.submit(Ms_Calphad, **composition_dict))
                futures["keys"].append("TD")
            if "NN" in models:
                futures["futures"].append(
                    pool.submit(Ms_ML.inference_vector, composition_vec)
                )
                futures["keys"].append("NN")
            if "EM" in models:
                futures["futures"].append(pool.submit(Ms_Ingber, **composition_dict))
                futures["keys"].append("EM")
            if "AG" in models:
                futures["futures"].append(pool.submit(Agrawal, **composition_dict))
                futures["keys"].append("AG")

        for _ in as_completed(futures["futures"]):
            prog.update(1)
        prog.close()

        id = 0
        for e1x in range(len(e1_rng)):
            for _ in models:
                Ms[futures["keys"][id]][e1x] = futures["futures"][id].result()
                id += 1
        for key in models:
            Ms[key] = remove_outlier(Ms[key])

        measured, diff, hover_text = import_filter_measured(
            "data/MsDatabase_2022_complete.csv", e1, None, threshold_measured_data_display, df, study["ref"]
        )

        for idx, key in enumerate(models):
            fig.add_trace(
                go.Scatter(
                    x=e1_rng,
                    y=Ms[key],
                    name=shortcut_to_long[key],
                    line=dict(color=colors[idx], width=4),
                    showlegend=False,
                    legendgroup="legend_" + str(idx),
                ),
                row=1,
                col=sx + 1,
            )
            if sx == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        name=shortcut_to_long[key],
                        line=dict(color=colors[idx], width=4),
                        showlegend=True,
                        legendgroup="legend_" + str(idx),
                    ),
                    row=1,
                    col=sx + 1,
                )

        fig.add_trace(
            go.Scatter(
            x=measured[e1["element"]],
            y=measured["Ms"],
            mode="markers",
            name="Measured data",
            showlegend=(sx == 0),
            marker=dict(
                size=8, 
                color=diff.values,
                colorscale="gray", 
                reversescale=False,
            ),
            hoverinfo="text",
            text=hover_text,
            legendgroup="legend_data",
            ),
            row=1,
            col=sx + 1,
        )
        fig.update_xaxes(
            title_text=study["e1"]["element"] + " [wt%]", row=1, col=sx + 1
        )
        fig.update_yaxes(title_text="Ms [K]", row=1, col=sx + 1)
        
    return fig


def range_study_2D(studies, models=["NN", "EM", "TD"], df=None, Ms_ML=None, threshold_measured_data_display=0.03):
    """
    Study the dependence of Martensite start temperature (Ms) on the composition of a steel alloy in a 2D parameter space.

    This function conducts a study to analyze how the Martensite start temperature (Ms) of a steel composition
    depends on the variation in the composition of two specific elements (e1 and e2) in a 2D parameter space.
    It can use different computational models (e.g., thermodynamic, empirical, neural network) to predict Ms
    and compare the results with measured data.

    Args:
        studies (list of dict): A list of dictionaries, each representing a study with specific parameters.
            Each study dictionary should include:
            - "e1" (dict): Information about the composition range of the first element, including:
                - "element" (str): The symbol of the first element (e.g., "C", "N").
                - "min" (float): The minimum composition value for the first element.
                - "max" (float): The maximum composition value for the first element.
                - "sample_points" (int): The number of sample points within the range for the first element.
            - "e2" (dict): Information about the composition range of the second element, including:
                - "element" (str): The symbol of the second element (e.g., "Cr", "Mn").
                - "min" (float): The minimum composition value for the second element.
                - "max" (float): The maximum composition value for the second element.
                - "sample_points" (int): The number of sample points within the range for the second element.
            - "ref" (dict, optional): An optional reference composition dictionary.

        models (list, optional): A list of models to use for computing Ms (default: ["NN", "EM", "TD"]).
        df (DataFrame, optional): An optional DataFrame for plotting additional data processing (default: None).
        Ms_ML (object, optional): An instance of a machine learning model for Ms prediction (default: None).
        threshold_measured_data_display (float, optional): The threshold for filtering measured data for display

    Returns:
        plotly.graph_objs.Figure: A Plotly figure containing 3D surface plots displaying Ms dependence on
        the composition of the two specified elements (e1 and e2) for each study. The computed results
        from different models are shown and compared with measured data.

    Notes:
        - The function uses parallel processing to efficiently compute Ms for a grid of composition values in 2D space.
        - The results are visualized as 3D surface plots for easy comparison.

    Example:
        # Define study parameters
        studies = [
            {
                "e1": {"element": "C", "min": 0.0, "max": 1.5, "sample_points": 16},
                "e2": {"element": "Cr", "min": 0.0, "max": 15.0, "sample_points": 16},
            },
            {
                "e1": {"element": "C", "min": 0.0, "max": 1.5, "sample_points": 16},
                "e2": {"element": "Mn", "min": 0.0, "max": 5.0, "sample_points": 16},
                "ref": composition_dict,
            },
            {
                "e1": {"element": "N", "min": 0.0, "max": 1.5, "sample_points": 16},
                "e2": {"element": "V", "min": 0.0, "max": 5.0, "sample_points": 16},
            },
        ]

        # Compute and visualize 2D Ms dependence
        fig = range_study_2D(studies, models=["NN", "TD"])
        fig.show()
        fig.write_html("plot_2D.html")
    """
    pool = ProcessPoolExecutor(max_workers=cpu_count())
    if Ms_ML is None:
        Ms_ML = DeployModel.load_from_checkpoint(
            "src/checkpoint/checkpoint", map_location="cpu"
        )

    fig = make_subplots(
        rows=1,
        cols=len(studies),
        specs=[[{"is_3d": True} for _ in range(len(studies))]],
    )
    colorscale = ["#ff7f0e", "#1f77b4", "#2ca02c", "#4a6528"]

    for sx, study in enumerate(studies):
        e1 = study["e1"]
        e2 = study["e2"]
        if "ref" not in study:
            study["ref"] = None
        e1_rng = np.linspace(e1["min"], e1["max"], e1["sample_points"])
        e2_rng = np.linspace(e2["min"], e2["max"], e2["sample_points"])
        Ms = {
            model: np.zeros([e1["sample_points"], e2["sample_points"]], float)
            for model in models
        }
        futures = {"futures": [], "keys": []}
        prog = tqdm(
            None,
            total=e1["sample_points"] * e2["sample_points"] * len(models),
            desc="Study " + str(sx + 1) + "/" + str(len(studies)),
        )
        for e1_val in e1_rng:
            for e2_val in e2_rng:
                composition_dict, composition_vec = comp_dict_vec_2D(
                    e1, e2, e1_val, e2_val, Ms_ML.order, study["ref"]
                )
                if "TD" in models:
                    futures["futures"].append(
                        pool.submit(Ms_Calphad, **composition_dict)
                    )
                    futures["keys"].append("TD")
                if "NN" in models:
                    futures["futures"].append(
                        pool.submit(Ms_ML.inference_vector, composition_vec)
                    )
                    futures["keys"].append("NN")
                if "EM" in models:
                    futures["futures"].append(
                        pool.submit(Ms_Ingber, **composition_dict)
                    )
                    futures["keys"].append("EM")
                if "AG" in models:
                    futures["futures"].append(
                        pool.submit(Agrawal, **composition_dict)
                    )
                    futures["keys"].append("AG")

        for _ in as_completed(futures["futures"]):
            prog.update(1)
        prog.close()

        id = 0
        for e1x in range(len(e1_rng)):
            for e2x in range(len(e2_rng)):
                for _ in models:
                    Ms[futures["keys"][id]][e1x, e2x] = futures["futures"][id].result()
                    id += 1
        for key in models:
            Ms[key] = remove_outlier(Ms[key])

        measured, diff, hover_text = import_filter_measured(
            "data/MsDatabase_2022_complete.csv", e1, e2, threshold_measured_data_display, df, study["ref"]
        )

        X, Y = np.meshgrid(e1_rng, e2_rng)
        n = len(colorscale)
        colors = [
            np.ones(shape=[e1["sample_points"], e2["sample_points"]])
            * (i / max((n - 1), 1))
            for i in range(n)
        ]

        for idx, key in enumerate(models):
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Ms[key].transpose(),
                    name=shortcut_to_long[key],
                    cmin=0,
                    cmax=1,
                    showscale=False,
                    colorscale=colorscale,
                    opacity=0.9,
                    surfacecolor=colors[idx],
                    legendgroup="legend_" + str(idx),
                ),
                row=1,
                col=sx + 1,
            )
            if sx == 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=[None],
                        y=[None],
                        z=[None],
                        mode="markers",
                        name=shortcut_to_long[key],
                        showlegend=True,
                        marker=dict(size=8, color=colorscale[idx]),
                        legendgroup="legend_" + str(idx),
                    ),
                    row=1,
                    col=sx + 1,
                )

        fig.add_trace(
            go.Scatter3d(
            x=measured[e1["element"]],
            y=measured[e2["element"]],
            z=measured["Ms"],
            mode="markers",
            name="Measured data",
            showlegend=(sx == 0),
            marker=dict(
                size=8, 
                color=diff.values,
                colorscale="gray", 
                reversescale=False,
            ),
            hoverinfo="text",
            text=hover_text,
            legendgroup="legend_data",
            ),
            row=1,
            col=sx + 1,
        )

        scene_layout = dict(
            xaxis=dict(title=study["e1"]["element"] + " [wt%]"),
            yaxis=dict(title=study["e2"]["element"] + " [wt%]"),
            zaxis=dict(title="Ms [K]"),
            bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(**{f"scene{sx+1}": scene_layout})

    return fig
