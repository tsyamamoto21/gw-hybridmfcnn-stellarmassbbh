"""
psd.py
"""
from scipy.interpolate import interp1d
import numpy as np


def PSD_LIGO_O1_analytic():
    """ Taken from LIGO's tutorial """
    return lambda freqs: (1.e-22 * (18. / (0.1 + freqs))**2)**2 + 0.7e-23**2 + ((freqs / 2000.) * 4.e-23)**2


def PSD_LIGO_O1_Hanford():
    """Taken from GWOSC"""
    data = np.genfromtxt("psdfiles/LIGO_O1H.txt")
    return interp1d(data[:, 0], data[:, 1]**2.0, fill_value="extrapolate")


def PSD_aLIGOdesign():
    """
    LIGO-T1800044-v5
    """
    data = np.genfromtxt("psdfiles/aLIGOdesign.txt")
    return interp1d(data[:, 0], data[:, 1]**2.0, fill_value="extrapolate")


def PSD_DECIGO():
    """
    Yagi & Seto, arXiv:1101.3940
    """
    fp = 7.36
    return lambda freqs: 7.05e-48 * (1 + (freqs / fp)**2.) + 4.8e-51 * (freqs**(-4.)) / (1 + (freqs / fp)**2.) + 5.33e-52 * (freqs**(-4.))


def PSD_BDECIGO():
    """
    Eq.(20) of Isoyama etal. (arXiv:1802.06977)
    """
    S0 = 4.040e-46
    return lambda f: S0 * (1. + 1.584e-2 * (f**(-4.)) + 1.584e-3 * (f**2.))
