import pickle
from scipy.interpolate import UnivariateSpline
from scipy.optimize import newton
import numpy as np
import matplotlib.pyplot as plt
from src.utilities_steel import weight_pct2mol

class Ms_Ghosh_Olson:
    """
    A class for calculating the martensite start temperature (Ms) based on the Ghosh-Olson model.
    This is meant to be an interface for different thermodynamic computation packages, such as pycalphad.

    Parameters:
    - data (dict): A dictionary containing the 'T' and 'dG' values used for calculations.

    Attributes:
    - Parameters (numpy.recarray): A structured array containing parameters for different elements.
    - K1 (float): Constant parameter for the calculation.
    - p (float): Exponent parameter for the calculation.
    - q (float): Exponent parameter for the calculation.
    - Tmu (float): Temperature parameter for the calculation.
    - W0FE (float): Constant parameter for the calculation.
    - T0 (float): Reference temperature at which the dG value is the smallest.
    - dG_interp (UnivariateSpline): A spline function representing the dG values as a function of temperature.

    Methods:
    - Ms(composition): Calculates the martensite start temperature (Ms) for a given composition.

    """
    b_plot = False
    if b_plot:
        plt.ion()
        T_rng = np.linspace(0,512,128)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc =  plt.plot(T_rng,T_rng,marker='o',ls='')
        li = plt.plot(T_rng,T_rng)
        li2 = plt.plot(T_rng,T_rng)
        ax.set_xlabel('T [K]')
        ax.set_ylabel('G [J]')
        plt.legend(['Calphad','Spline','GO-dG-Critical'])

    def __init__(self, data=None, parameters=None, dbf=None):
        """
        Constructor: Initializes the Ms_Ghosh_Olson object with the provided data.

        Parameters:
        - data (dict): A dictionary containing the 'T' and 'dG' values used for calculations. 'dG' is typically the result of CALPHAD computations.
        - composition as dict in wt%

        """
        self.Parameters = {
            "K1": 1010.0,
            "C": 4009.0,
            "SI": 1879.0,
            "MN": 1980.0,
            "NI": 172.0,
            "MO": 1418.0,
            "CR": 1868.0,
            "V": 1618.0,
            "CU": 752.0,
            "W": 714.0,
            "NB": 1653.0,
            "N": 3097.0,
            "CO": -352.0,
            "TI": 1473.0,
            "AL": 280.0,
        }
        if parameters is not None:
            if isinstance(parameters, dict): 
                self.Parameters = parameters
            else:
                assert(len(parameters)==len(self.Parameters))
                for id, key in enumerate(self.Parameters):
                    self.Parameters[key] = parameters[id]
        self.dbf = dbf
        if data is not None:
            self._compute_spline(data)

    def load_parameters(self, path, opt_key):
        with open(path,'rb') as file:
            data = pickle.load(file)
        for id, key in enumerate(self.Parameters):
            self.Parameters[key] = data[opt_key]['x']['x'][id]


    def _compute_dG_crit(self, composition):
        """
        Computes the critical Energy for phase transition for the given composition.

        Parameters:
        - composition (dict): A dictionary containing the element compositions.

        """

        Wmui = 0
        Wmuj = 0
        Wmuk = 0
        Wmul = 0

        for element, concentration in composition.items():
            el = element.upper()
            if el in self.Parameters.keys():
                Kmu = self.Parameters[el]
                if el in ["C", "N"]:
                    Wmui += (np.sqrt(concentration) * Kmu) ** 2
                elif el in ["CR", "MN", "MO", "NB", "SI", "TI", "V"]:
                    Wmuj += (np.sqrt(concentration) * Kmu) ** 2
                elif el in ["AL", "CU", "NI", "W"]:
                    Wmuk += (np.sqrt(concentration) * Kmu) ** 2
                elif el in ["CO"]:
                    Wmul += np.sqrt(concentration) * Kmu

        self.dG_crit = -(self.Parameters["K1"] + np.sqrt(Wmui) + np.sqrt(Wmuj) + np.sqrt(Wmuk) + Wmul)
        
        if Ms_Ghosh_Olson.b_plot:
            composition_str = ', '.join(f"'{key}': {round(val, 2)}" for key, val in composition.items())
            Ms_Ghosh_Olson.ax.set_title(composition_str)

    def _compute_spline(self, data):
        """
        Sets the dG data and performs sorting and spline interpolation of dG over T.

        Parameters:
        - data (dict): A dictionary containing the 'T' and 'dG' values used for calculations.

        """
        b_nan = ~np.isnan(data['dG'])
        dG = np.rec.array(
            [(t, dG) for t, dG in zip(data["T"][b_nan], data["dG"][b_nan])], names=["T", "dG"]
        )
        dG.sort(order="T")
        self.E_ath_fr_spline = UnivariateSpline(dG["T"], dG["dG"], s=1)
        self.E_ath_fr_spline_prime = self.E_ath_fr_spline.derivative()

        if Ms_Ghosh_Olson.b_plot:
            T_rng = np.linspace(dG["T"].min(),dG["T"].max(),128)
            dg_crit = self.Parameters["K1"] + self.dG_crit
            y_rng = np.append(dG["dG"],dg_crit)
            Ms_Ghosh_Olson.sc[0].set_xdata(dG["T"])
            Ms_Ghosh_Olson.sc[0].set_ydata(dG["dG"])
            Ms_Ghosh_Olson.li[0].set_xdata(T_rng)
            Ms_Ghosh_Olson.li[0].set_ydata(self.E_ath_fr_spline(T_rng))
            Ms_Ghosh_Olson.li2[0].set_xdata([T_rng[0],T_rng[-1]])
            Ms_Ghosh_Olson.li2[0].set_ydata([dg_crit, dg_crit])
            Ms_Ghosh_Olson.ax.set_xlim(min(T_rng),max(T_rng))
            Ms_Ghosh_Olson.ax.set_ylim(min(y_rng)-100,max(y_rng)+100)
            Ms_Ghosh_Olson.fig.canvas.draw()
            Ms_Ghosh_Olson.fig.canvas.flush_events()

    def _compute_delta_critical_dG(self, T):
        """
        Calculates the critical dG value for a given temperature.

        Parameters:
        - T (float): Temperature at which to calculate the critical dG.
        - dG_interp (UnivariateSpline): Spline function representing the dG values.

        Returns:
        - crit_DG (float): Critical dG value at the given temperature.

        """
        delta_crit_DG = self.E_ath_fr_spline(T) - self.dG_crit

        return delta_crit_DG

    def Ms(self, T_initial=500, composition=None, data=None):
        """
        Calculates the martensite start temperature (Ms) for the given composition.

        Parameters:
        - T_initial (float): Initial guess of Ms Temperature
        - composition (dict): A dictionary containing the element compositions.

        Returns:
        - Ms (float): Martensite start temperature.

        """
        if data is not None:
            self._compute_spline(data)

        self._compute_dG_crit(weight_pct2mol(composition, self.dbf))
        try:
            self.T_initial = T_initial
            Ms = newton(self._compute_delta_critical_dG, self.T_initial,fprime=self.E_ath_fr_spline_prime, maxiter=100)
            return Ms.item()
        except:
            # plt.plot(self._compute_critical_dG(np.linspace(0,5000,128),self.E_ath_fr_spline))
            return np.nan