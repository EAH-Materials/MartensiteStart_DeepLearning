# -*- coding: utf-8 -*-
from src.RangeCompute import range_study

if __name__ == "__main__":
    studies = [
        {'e1':{"element": "C", "min": 0.0, "max": 1.5, "sample_points": 64}, 'e2':{"element": "Mn", "min": 0.0, "max": 15.0, "sample_points": 64}},
        {'e1':{"element": "C", "min": 0.0, "max": 1.5, "sample_points": 64}, 'e2':{"element": "Cr", "min": 0.0, "max": 15.0, "sample_points": 64}},
        {'e1':{"element": "C", "min": 0.0, "max": 1.5, "sample_points": 64}, 'e2':{"element": "V", "min": 0.0, "max": 5.0, "sample_points": 64}},
               ]
    fig = range_study(studies)
    fig.show()
    fig.write_html("plot.html")
