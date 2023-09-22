from math import exp

try:
    from pycalphad import variables as v
    no_pycalphad = False
except ModuleNotFoundError:
    no_pycalphad = True
    pass

atomicweights = {  # [g/mol]
    "FE": 55.845,  # Iron
    "C": 12.011,  # Carbon
    "MN": 54.938,  # Manganese
    "SI": 28.085,  # Silicon
    "CR": 51.996,  # Chromium
    "NI": 58.693,  # Nickel
    "MO": 95.950,  # Molybdenum
    "V": 50.942,  # Vanadium
    "CO": 58.933,  # Cobalt
    "AL": 26.982,  # Aluminum
    "W": 183.84,  # Tungsten
    "CU": 63.546,  # Copper
    "NB": 92.906,  # Niobium
    "TI": 47.867,  # Titanium
    "B": 10.81,  # Boron
    "N": 14.007,  # Nitrogen
}


def weight_pct2mol(composition, dbf=None):
    """
    Convert weight percent composition to mole fractions in a steel alloy.

    Parameters:
    - composition (dict): A dictionary specifying the weight percent composition of the alloy.
                          The keys are element symbols, and the values are the weight percent fractions.

    Returns:
    - mole_fractions (dict): A dictionary containing the mole fractions of each element in the alloy.
                             The keys are element symbols, and the values are the corresponding mole fractions.

    Notes:
    - The function assumes the basis of iron (Fe) as 100% unless explicitly specified.
    - The considered elements in the calculation include:
        - Fe (Iron)
        - C (Carbon)
        - Mn (Manganese)
        - Si (Silicon)
        - Cr (Chromium)
        - Ni (Nickel)
        - Mo (Molybdenum)
        - V (Vanadium)
        - Co (Cobalt)
        - Al (Aluminum)
        - W (Tungsten)
        - Cu (Copper)
        - Nb (Niobium)
        - Ti (Titanium)
        - B (Boron)
        - N (Nitrogen)
      If an element is not specified in the composition, its weight percent fraction is assumed to be 0%.
    - The atomic weights of the elements are based on standard values.

    Raises:
    - ValueError: If an element symbol in the composition is not found in the atomicweights dictionary.
    - AssertionError: If Fe is explicitly specified and the composition does not sum up to 100% (with a tolerance of 0.5%).
    """

    if dbf is None or no_pycalphad:
        element_compositions = {}
        total_composition = 0.0

        for element, fraction in composition.items():
            if element.upper() not in atomicweights:
                raise ValueError(f"Atomic weight not found for element: {element}")
            element_compositions[element.upper()] = fraction / 100
            total_composition += element_compositions[element.upper()]

        if "FE" not in element_compositions:
            element_compositions["FE"] = 1.0 - total_composition
            total_composition = 1
        else:
            assert abs(total_composition - 1.0) <= 0.005

        mole_fractions = {}
        mole_sum = 0
        for element, fraction in element_compositions.items():
            mole_fractions[element] = element_compositions[element] / atomicweights[element]
            mole_sum += mole_fractions[element]

        for element, fraction in element_compositions.items():
            mole_fractions[element] = mole_fractions[element] / mole_sum

        return mole_fractions
    else:
        composition_wt_v = {v.W(key): value / 100 for key, value in composition.items()}
        composition_mol_v = v.get_mole_fractions(composition_wt_v, "FE", dbf)
        composition_mol_v = {
            str(key)[2:]: value for key, value in composition_mol_v.items()
        }
        composition_mol_v["FE"] = 1.0 - sum(composition_mol_v.values())

        return composition_mol_v

def mol2weight_pct(composition, dbf=None):
    """
    Convert mole fraction composition to weight percent composition in a steel alloy.

    Parameters:
    - composition (dict): A dictionary specifying the mole fractions of each element in the alloy.
                          The keys are element symbols, and the values are the mole fractions.

    Returns:
    - weight_percent (dict): A dictionary containing the weight percent fractions of each element in the alloy.
                             The keys are element symbols, and the values are the corresponding weight percent fractions.

    Notes:
    - The function assumes the basis of iron (Fe) as 100% unless explicitly specified.
    - The considered elements in the calculation include:
        - Fe (Iron)
        - C (Carbon)
        - Mn (Manganese)
        - Si (Silicon)
        - Cr (Chromium)
        - Ni (Nickel)
        - Mo (Molybdenum)
        - V (Vanadium)
        - Co (Cobalt)
        - Al (Aluminum)
        - W (Tungsten)
        - Cu (Copper)
        - Nb (Niobium)
        - Ti (Titanium)
        - B (Boron)
        - N (Nitrogen)
      If an element is not specified in the composition, its mole fraction is assumed to be 0.

    Raises:
    - ValueError: If an element symbol in the composition is not found in the atomicweights dictionary.
    """

    if dbf is None or no_pycalphad:
        total_mole_fraction = sum(composition.values())

        if "FE" not in composition:
            composition["FE"] = 1.0 - total_mole_fraction
            total_mole_fraction = 1.0
        else:
            assert (
                abs(total_mole_fraction - 1.0) <= 0.005
            ), "Composition must add up to 100% (within a tolerance of 0.5%) when Fe is explicitly specified."

        weight_percent = {}
        total_atomicweight = 0.0
        for element, mole_fraction in composition.items():
            if element.upper() not in atomicweights:
                raise ValueError(f"Atomic weight not found for element: {element}")
            total_atomicweight += atomicweights[element.upper()] * mole_fraction

        for element, mole_fraction in composition.items():
            weight_fraction = (
                atomicweights[element.upper()] * mole_fraction / total_atomicweight
            )
            weight_percent[element.upper()] = weight_fraction * 100.0
        return weight_percent
    else:
        composition_mol_v = {
        v.MoleFraction(key): value for key, value in composition.items()
        }
        composition_wt_v = v.get_mass_fractions(composition_mol_v, "CR", dbf)
        composition_wt_v = {
            str(key)[2:]: value * 100.0 for key, value in composition_wt_v.items()
        }
        return composition_wt_v


def Ms_Ingber(**kwargs):
    inputs = {k.upper(): v for k, v in kwargs.items()}
    """Compute Ms with an empirical model"""
    C = inputs.get("C", 0.0)
    Mn = inputs.get("MN", 0.0)
    Si = inputs.get("SI", 0.0)
    Cr = inputs.get("CR", 0.0)
    Mo = inputs.get("MO", 0.0)
    Ni = inputs.get("NI", 0.0)
    Al = inputs.get("AL", 0.0)
    Co = inputs.get("CO", 0.0)

    return (
        round(
            530.2
            - 290.3 * C
            - 35.5 * Mn
            - 6.8 * Si
            - 17.2 * Ni
            - 20.8 * Cr
            - 10.4 * Mo
            + 7.1 * Al
            + 4.8 * Co
            - 75 * (1 - exp(-0.96 * C))
        )
        + 273.15
    )
