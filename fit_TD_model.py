# %% Import stuff
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pickle
from src.Ghosh_Olson import Ms_Ghosh_Olson
from src.utilities_steel import Ms_Ingber
from src.MS_Pycalphad import get_point, dbf
from pycalphad import calculate

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import os


meas_data = pd.read_csv(os.path.join("data", "MsDatabase_2022.csv"))

methods = ['Powell', 'Nelder-Mead','BFGS', 'SLSQP', ]

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
li_p = {method: plt.plot(0,0) for method in methods}
f_x = {method: [] for method in methods}
ax.set_xlabel('Iteration')
ax.set_ylabel('f(x)')
plt.legend(methods)

global max_x, max_y, min_y 
max_x = -float('inf') 
min_y = float('inf')
max_y = -float('inf')

def loss_fcn(x0, Ts, dGs, MS_meas, method=None):
    global max_x, max_y, min_y
    ms_GO = Ms_Ghosh_Olson(parameters=x0, dbf=dbf)
    MS_comp = np.zeros_like(MS_meas)
    for id in range(len(meas_data)):
        T_guess = Ms_Ingber(**meas_data.iloc[id])
        MS_comp[id] = ms_GO.Ms(T_guess, {**meas_data.iloc[id]}, {'T':Ts,'dG':dGs[id,:]})

    b_nan = ~np.isnan(MS_comp)
    penalty = np.sum(np.isnan(MS_comp))
    d_ms = np.mean(np.abs(MS_comp[b_nan]-MS_meas[b_nan])) + penalty
    # d_ms = np.mean(np.sqrt((MS_comp[b_nan]-MS_meas[b_nan])**2)) 
    if method is not None:
        f_x[method].append(d_ms)
        for _, vals in f_x.items():
            if vals:
                max_x = max(max_x, len(vals))
                min_y = min(min_y, min(vals))
                max_y = max(max_y, max(vals))

        li_p[method][0].set_xdata(np.linspace(1,len(f_x[method]),len(f_x[method])))
        li_p[method][0].set_ydata(f_x[method])
        ax.set_xlim(1,max_x+1)
        ax.set_ylim(min_y-1, max_y+1)
        fig.canvas.draw()
        fig.canvas.flush_events()
    return d_ms

def dG_Calphad(Ts, **kwargs):
    inputs = {k.upper(): v for k, v in kwargs.items()}
    Ms = inputs.pop("MS")
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
    

    # Compute the 'points'-array, specific to pycalphad-convention and compute the Gibbs energies
    site_fractions = get_point(
        inputs
    )

    # Compute Gibbs energies at ambient pressure
    td_res = calculate(
        dbf, comps, phases, T=Ts, P=101325, points=site_fractions
    )

    # Get deltaG - the point where dG is zero marks the equilibrium condition (i.e. Austenite and Martensite are thermodynamically equally probable)
    dG = (
        td_res.GM[..., td_res.phase_indices["BCC_A2"]]
        - td_res.GM[..., td_res.phase_indices["FCC_A1"]]
    ).squeeze()

    return dG, Ms

# %%Compute all dGs
if not os.path.exists("data/dGs.npy"):
    Ts = np.linspace(273.15, 1400, 16)
    dGs = np.zeros((len(meas_data),len(Ts)))
    Mss = np.zeros((len(meas_data),1))
    pool = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []
    prog = tqdm(None, total=len(meas_data))
    for id in range(len(meas_data)):
        futures.append(pool.submit(dG_Calphad, Ts, **meas_data.iloc[id]))

    for future in as_completed(futures):
        prog.update(1)
    prog.close()

    for id in range(len(meas_data)):
        dGs[id,:],Mss[id,:] = futures[id].result()

    np.save("data/dGs.npy",dGs)
    np.save("data/Mss.npy",Mss)
    np.save("data/Ts.npy",Ts)

# %% Fit
meas_data.pop("Ms")
dGs = np.load("data/dGs.npy")
Mss = np.load("data/Mss.npy")
Ts = np.load("data/Ts.npy")

x0 = [1010.0,4009.0,1879.0,1980.0,172.0,1418.0,1868.0,1618.0,752.0,714.0,1653.0,3097.0,-352.0,1473.0,280.0]

res = {method: [] for method in methods}
def optimize_method(method):
    args = (Ts, dGs, Mss, method)
    result = minimize(loss_fcn, x0, args=args, method=method, tol=1e-2, options={'maxiter':400,'xrtol':0.1})
    return method, result, result.fun

for method in methods:
    method, result, val = optimize_method(method)
    res[method] = {'x':result,'val':val,'history':f_x[method]}
    print(f'{method}: f(x): {val}   x: {result.x}')

    with open('data/optimization_results_'+method+'.pickle', 'wb') as file:
        pickle.dump(res, file)


with open('data/optimization_results.pickle', 'wb') as file:
    pickle.dump(res, file)

plt.savefig('Fitting.svg')

