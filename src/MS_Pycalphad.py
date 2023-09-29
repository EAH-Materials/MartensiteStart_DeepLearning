import pickle
from pycalphad import Database, calculate, variables as v
from src.utilities_steel import Ms_Ingber, weight_pct2mol
from src.Ghosh_Olson import Ms_Ghosh_Olson
import numpy as np
from collections import OrderedDict
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# Preload database
dbf = Database(
    os.path.join(os.path.dirname(__file__), "mc_fe_v2.059.pycalphad.tdb")
)  # Matcalc iron
warnings.resetwarnings()
 
with open('data/optimization_results.pickle','rb') as file:
    fitted_go_prms = pickle.load(file)['Nelder-Mead']['x']['x'] #'Nelder-Mead' 'SLSQP' 'Powell' 'BFGS'
    
def get_point(composition):
    """
    Generate site fractions for a given chemical composition in Fe-alloys.

    Parameters:
        composition (dict): A dictionary with keys representing chemical elements and their corresponding fractional values in wt%.

    Returns:
        numpy.ndarray: A 2D numpy array containing site fractions. Each row represents a different combination of element fractions,
                        and each column corresponds to an element, with the last two columns representing 'C' and 'Vacancy' fractions on the interstitial
                        sublattice site and the former columns account for all elements on the 'Fe'-sublattice in alphabetical order, in accordance
                        with the pycalphad convention.

    Note:
        This function calculates the site fractions for a specific composition.
        The 'Fe' fraction is calculated as the complement to ensure that the total
        sum of fractions for each combination adds up to 1.0.
    """

    # get composition in moles
    comp = weight_pct2mol(composition, dbf)

    # Sort the elements alphabetically
    sorted_elements = sorted(comp.keys())

    # Create arrays for each site in sorted alphabetical order
    site_1 = OrderedDict({})
    site_2 = OrderedDict({})
    for element in sorted_elements:
        if element in ["C", "N", "B", "H"]:
            site_2[element] = comp[element]
        else:
            site_1[element] = comp[element]

    site_1_sum = np.sum(np.stack([site_1[key] for key in site_1], axis=-1), axis=-1)
    site_2_sum = 1.0 - site_1_sum
    
    res = {
        "FCC_A1": np.zeros((1, len(sorted_elements) + 1)),
        "BCC_A2": np.zeros((1, len(sorted_elements) + 1)),
    }

    ix = 0
    for element in sorted_elements:
        if element not in ["C", "N", "B", "H"]:
            res["FCC_A1"][:, ix] = site_1[element] / site_1_sum
            res["BCC_A2"][:, ix] = site_1[element] / site_1_sum
            ix += 1

    site_2_sum_r = (1-site_2_sum)
    if site_2_sum > 0.0:
        for element in sorted_elements:
            if element in ["C", "N", "B", "H"]:
                res["FCC_A1"][:, ix] = site_2[element] / site_2_sum_r
                res["BCC_A2"][:, ix] = (site_2[element] / site_2_sum_r) / 3.0
                ix += 1

    res["FCC_A1"][:, ix] = 1.0 - site_2_sum
    res["BCC_A2"][:, ix] = 1.0 - (site_2_sum / 3.0)
    return res


def ms_Calphad(T_guess=None, **kwargs):
    inputs = {k.upper(): v for k, v in kwargs.items()}

    comps = ["FE", "VA"]  # Elements to consider ('VA'=Vacancies)
    drop = []
    for e, va in inputs.items():
        if va > 0.0:
            comps.append(e)
        else:
            drop.append(e)
    for e in drop:
        inputs.pop(e)

    phases = ["FCC_A1", "BCC_A2"]  # Austenite and Martensite phases

    Ts = np.linspace(273.15, 1400, 16)  # Temperature range and sampling for finding Ms.
    # This sampling can be rather coarse, since the Gibbs energies
    # give continuous, smooth curves over T and easily be interpolated later

    # Compute the 'points'-array, specific to pycalphad-convention and compute the Gibbs energies
    site_fractions = get_point(inputs)

    # Compute Gibbs energies at ambient pressure
    td_res = calculate(dbf, comps, phases, T=Ts, P=101325, points=site_fractions)

    # Get deltaG - the point where dG is zero marks the equilibrium condition (i.e. Austenite and Martensite are thermodynamically equally probable)
    dG = (
        td_res.GM[..., td_res.phase_indices["BCC_A2"]]
        - td_res.GM[..., td_res.phase_indices["FCC_A1"]]
    )

    # Use the Ghosh-Olsen model to compute the energy needed to initiate phase transition. This shifts the dG-curve accordingly
    # and the root of this shifted curve defines the Ms-temperature. Initializing Ms_Ghosh_Olson computes
    # a spline interpolation of the dG-over-T curve
    ms_GO = Ms_Ghosh_Olson({"T": Ts, "dG": dG.squeeze()}, parameters=fitted_go_prms, dbf=dbf)
    
    # To find the root of the interpolated curve the Newton algorithm is used, which requires a starting point.
    # We use a computationally cheap empirical model for the initial guess of Ms
    if T_guess is None:
        T_guess = Ms_Ingber(**inputs)

    # Compute the critical dG and solve the root finding problem to obtain Ms
    return ms_GO.Ms(
        T_initial=T_guess,
        composition=inputs,
    )
