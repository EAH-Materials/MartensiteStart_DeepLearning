
import numpy as np
import pandas as pd

from tqdm import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import cpu_count

from src.DeployModel import DeployModel
from src.utilities_steel import Ms_Ingber
from src.MS_Pycalphad import ms_Calphad as Ms_Calphad

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def import_filter_measured(path, range_1, range_2=None, threshold=0.05, df=None):
    if df is None:
        df = pd.read_csv(path)
    rest_elements = ["C","Mn","Si","Cr","Ni","Mo","V","Co","Al","W","Cu","Nb","Ti","B","N",]
    rest_elements.remove(range_1["element"])
    if range_2 is not None:
        rest_elements.remove(range_2["element"])
        filtered_df = df[
            (df[range_1["element"]].between(range_1["min"], range_1["max"]))
            & (df[range_2["element"]].between(range_2["min"], range_2["max"]))
            & (df[rest_elements] < threshold).all(axis=1)
        ]
    else:
            filtered_df = df[
            (df[range_1["element"]].between(range_1["min"], range_1["max"]))
            & (df[rest_elements] < threshold).all(axis=1)
        ]
    return filtered_df

def comp_dict_vec_1D(e1, e1_val, order):
    composition_dict = {e1["element"]: e1_val}
    composition_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    composition_vec[order.index(e1["element"])] = e1_val
    return composition_dict, composition_vec

def comp_dict_vec_2D(e1, e2, e1_val, e2_val, order):
    composition_dict = {e1["element"]: e1_val, e2["element"]: e2_val}
    composition_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    composition_vec[order.index(e1["element"])] = e1_val
    composition_vec[order.index(e2["element"])] = e2_val
    return composition_dict, composition_vec

def remove_outlier(array, threshold=5):
    zscore = (array - np.nanmean(array))/np.nanstd(array)
    array[np.abs(zscore) > threshold] = np.nan
    array[array < 0.0] = np.nan
    return array

shortcut_to_long = {
    'EM': 'Empirical (Ingber)',
    'NN': 'Neural Network',
    'TD': 'Thermodynamic',
}

def range_study_1D(studies, models=['NN','EM','TD'], df=None, Ms_ML=None):
    pool = ProcessPoolExecutor(max_workers=cpu_count())
    if Ms_ML is None:
        Ms_ML = DeployModel.load_from_checkpoint("src/checkpoint/checkpoint", map_location="cpu")

    fig = make_subplots(rows=1, cols=len(studies))  
    colors = ["#ff7f0e", "#1f77b4", "#2ca02c"]

    for sx, study in enumerate(studies):
        e1 = study['e1']
        e1_rng = np.linspace(e1["min"], e1["max"], e1["sample_points"])
        Ms = {model: np.zeros([e1["sample_points"]], float) for model in models}
        futures = {'futures':[],'keys':[]}
        prog = tqdm(None, total=e1["sample_points"] * len(models),desc='Study '+str(sx+1)+'/'+str(len(studies)))
        for e1_val in e1_rng:
            composition_dict, composition_vec = comp_dict_vec_1D(e1, e1_val, Ms_ML.order)
            if 'TD' in models:
                futures['futures'].append(pool.submit(Ms_Calphad, **composition_dict))
                futures['keys'].append('TD')
            if 'NN' in models:
                futures['futures'].append(pool.submit(Ms_ML.inference_vector, composition_vec))
                futures['keys'].append('NN')
            if 'EM' in models:
                futures['futures'].append(pool.submit(Ms_Ingber, **composition_dict))
                futures['keys'].append('EM')

        for _ in as_completed(futures['futures']):
            prog.update(1)
        prog.close()

        id = 0
        for e1x in range(len(e1_rng)):
            for _ in models:
                Ms[futures['keys'][id]][e1x] = futures['futures'][id].result()
                id += 1
        for key in models:
            Ms[key] = remove_outlier(Ms[key])

        measured = import_filter_measured("data/MsDatabase_2022.csv", e1, None, 0.1, df)

        for idx, key in enumerate(models):
            fig.add_trace(
                go.Scatter(x=e1_rng, y=Ms[key], name=shortcut_to_long[key], line=dict(color=colors[idx], width=4), showlegend=False, legendgroup='legend_'+str(idx)),
                row=1, col=sx+1
            )
            if sx == 0:
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], name=shortcut_to_long[key], line=dict(color=colors[idx], width=4), showlegend=True, legendgroup='legend_'+str(idx)),
                    row=1, col=sx+1
                )

            
        fig.add_trace(
            go.Scatter(x=measured[e1["element"]], y=measured["Ms"], mode="markers", name="Measured data", showlegend=(sx == 0), marker=dict(size=8, color="black"),legendgroup='legend_data'),
            row=1, col=sx+1
        )
        fig.update_xaxes(title_text=study['e1']["element"] + " [wt%]",row=1, col=sx+1)
        fig.update_yaxes(title_text="Ms [K]",row=1, col=sx+1)

    fig.update_layout(
        legend=dict(
            font=dict(size=30)
        )
    )
    return fig  

def range_study_2D(studies, models=['NN','EM','TD'], df=None, Ms_ML=None):
    pool = ProcessPoolExecutor(max_workers=cpu_count())
    if Ms_ML is None:
        Ms_ML = DeployModel.load_from_checkpoint("src/checkpoint/checkpoint", map_location="cpu")

    fig = make_subplots(rows=1, cols=len(studies), specs = [[{'is_3d': True} for _ in range(len(studies))]])  
    colorscale = ["#ff7f0e", "#1f77b4", "#2ca02c"]

    for sx, study in enumerate(studies):
        e1 = study['e1']
        e2 = study['e2']
        e1_rng = np.linspace(e1["min"], e1["max"], e1["sample_points"])
        e2_rng = np.linspace(e2["min"], e2["max"], e2["sample_points"])
        Ms = {model: np.zeros([e1["sample_points"], e2["sample_points"]], float) for model in models}
        futures = {'futures':[],'keys':[]}
        prog = tqdm(None, total=e1["sample_points"] * e2["sample_points"] * len(models),desc='Study '+str(sx+1)+'/'+str(len(studies)))
        for e1_val in e1_rng:
            for e2_val in e2_rng:
                composition_dict, composition_vec = comp_dict_vec_2D(e1, e2, e1_val, e2_val, Ms_ML.order)
                if 'TD' in models:
                    futures['futures'].append(pool.submit(Ms_Calphad, **composition_dict))
                    futures['keys'].append('TD')
                if 'NN' in models:
                    futures['futures'].append(pool.submit(Ms_ML.inference_vector, composition_vec))
                    futures['keys'].append('NN')
                if 'EM' in models:
                    futures['futures'].append(pool.submit(Ms_Ingber, **composition_dict))
                    futures['keys'].append('EM')

        for _ in as_completed(futures['futures']):
            prog.update(1)
        prog.close()

        id = 0
        for e1x in range(len(e1_rng)):
            for e2x in range(len(e2_rng)):
                for _ in models:
                    Ms[futures['keys'][id]][e1x, e2x] = futures['futures'][id].result()
                    id += 1
        for key in models:
            Ms[key] = remove_outlier(Ms[key])

        measured = import_filter_measured("data/MsDatabase_2022.csv", e1, e2, 0.1, df)

        X, Y = np.meshgrid(e1_rng, e2_rng)
        n = len(colorscale)  
        colors = [np.ones(shape=[e1["sample_points"], e2["sample_points"]]) * (i / max((n - 1),1)) for i in range(n)]

        for idx, key in enumerate(models):
            fig.add_trace(
                go.Surface(x=X, y=Y, z=Ms[key].transpose(), name=shortcut_to_long[key], cmin=0, cmax=1, showscale=False, colorscale=colorscale, opacity=0.9, surfacecolor=colors[idx],legendgroup='legend_'+str(idx)),
                row=1, col=sx+1
            )
            if sx == 0:
                fig.add_trace(
                    go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", name=shortcut_to_long[key], showlegend=True, marker=dict(size=8, color=colorscale[idx]),legendgroup='legend_'+str(idx)),
                    row=1, col=sx+1
                )

        fig.add_trace(
            go.Scatter3d(x=measured[e1["element"]], y=measured[e2["element"]], z=measured["Ms"], mode="markers", name="Measured data", showlegend=(sx == 0), marker=dict(size=8, color="black"),legendgroup='legend_data'),
            row=1, col=sx+1
        )

        scene_layout = dict(
            xaxis=dict(title=study['e1']["element"] + " [wt%]"),
            yaxis=dict(title=study['e2']["element"] + " [wt%]"),
            zaxis=dict(title="Ms [K]")
        )
        fig.update_layout(**{f"scene{sx+1}": scene_layout})

    fig.update_layout(
        legend=dict(
            font=dict(size=30)
        )
    )
    return fig  