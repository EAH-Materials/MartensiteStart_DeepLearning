# -*- coding: utf-8 -*-
from src.RangeCompute import range_study_1D, range_study_2D
from src.MS_Pycalphad import ms_Calphad

if __name__ == "__main__":
    
    # Single value
    # composition_dict = {'C':0.05}
    # T_TD = ms_Calphad(**composition_dict)

    # Range study 1D
    studies = [
        {'e1':{"element": "C", "min": 0.0, "max": 1.5, "sample_points": 16}},
        {'e1':{"element": "N", "min": 0.0, "max": 1.5, "sample_points": 16}},
               ]
    fig = range_study_1D(studies,models=['NN','EM','TD'])
    fig.show()
    fig.write_html("plot_1D.html")

    # Range study 2D
    # studies = [
    #     {'e1':{"element": "C", "min": 0.0, "max": 1.5, "sample_points": 16}, 'e2':{"element": "Mn", "min": 0.0, "max": 15.0, "sample_points": 16}},
    #     {'e1':{"element": "C", "min": 0.0, "max": 1.5, "sample_points": 16}, 'e2':{"element": "Al", "min": 0.0, "max": 5.0, "sample_points": 16}},
    #     # {'e1':{"element": "N", "min": 0.0, "max": 1.5, "sample_points": 16}, 'e2':{"element": "V", "min": 0.0, "max": 5.0, "sample_points": 16}},
    #            ]
    # fig = range_study_2D(studies,models=['NN','EM','TD'])
    # fig.show()
    # fig.write_html("plot_2D.html")

