"""
Author: Elias / Matthias
Date: 23/05/2025

Decompose the trajectory of the production run obtained in (a) in 6 blocks of equal time,
 and calculate for each block the adsorption Γ defined as Γ = (zw,1+zw,2)/2
 zw,1
 (ρ(z) − ρb)dz
 where ρb is the bulk density, i.e. the density defined at the box center z = (zw,1 +zw,2)/2.
 Calculate and report the corresponding statistical error for Γ
"""

import settings


def calc_adsorption(rho_z):
    """
    Calculate the adsorption based on the given parameters.

    Returns:
    float: The calculated adsorption value.
    """
    # summiere halbes array von null bis zur hälfte
    # weil das den Integralgrenzen von zw,1 bis (zw,1 + zw,2)/2 entspricht
    # sollte rho_z ungerade sein, dann wird len()-1/2 genommen
    rho_b = rho_z[len(rho_z) // 2 - 50 : len(rho_z) // 2 + 50].mean()
    adsorption = sum(rho_z[: len(rho_z) // 2] - rho_b) * settings.deltar
    return adsorption


def calc_statistical_error(adsorption_values):
    """
    Calculate the statistical error based on the given adsorption values.

    Returns:
    float: The calculated statistical error.
    """
    mean = sum(adsorption_values) / len(adsorption_values)
    variance = (
        sum((x - mean) ** 2 for x in adsorption_values)
        / (len(adsorption_values) - 1)  # N-1 im Nenner???
        / len(adsorption_values)  # N im Nenner ??
    )
    return variance**0.5
