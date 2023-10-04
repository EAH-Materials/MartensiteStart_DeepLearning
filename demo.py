# -*- coding: utf-8 -*-
from os.path import join
from src.RangeCompute import range_study_1D, range_study_2D
from src.MS_Pycalphad import ms_Calphad, Ms_Ingber
from src.DeployModel import DeployModel
from torch import device

if __name__ == "__main__":
    model = DeployModel.load_from_checkpoint(
        join("src", "checkpoint", "checkpoint"), map_location=device("cpu")
    )

    # Single values
    composition_dict = {"C": 0.55, "Si": 0.8}
    T_TD = ms_Calphad(**composition_dict)
    T_Em = Ms_Ingber(**composition_dict)
    T_NN = float(model.inference_dict(composition_dict))
    print([f'Thermodynamic: {T_TD.__round__()}K   Empirical: {T_Em.__round__()}K    ANN: {T_NN.__round__()}K'])

    # Range study 1D
    studies = [
        {"e1": {"element": "C", "min": 0.0, "max": 1.5, "sample_points": 16}},
        {
            "e1": {"element": "N", "min": 0.0, "max": 1.5, "sample_points": 16},
            "ref": composition_dict,
        },
    ]
    fig = range_study_1D(studies, models=["NN", "EM", "TD"], Ms_ML=model)
    fig.show()
    fig.write_html("plot_1D.html")

    # Range study 2D
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
    fig = range_study_2D(studies, models=["NN", "EM", "TD"], Ms_ML=model)
    fig.show()
    fig.write_html("plot_2D.html")
