"""
gw_parameters.py
"""
import numpy as np

def ISCO_radius(a):
    """ISCO radius
    Taken from Press, Teukolsky and *** 1972

    Args:
        a (float or numpy.ndarray): Spin parameter of black hole

    Returns:
        r (float or numpy.ndarray): ISCO radius
    """

    a2 = a*a
    Z1 = 1. + (1-a2)**(1./3.) * ((1+a)**(1./3.) + (1-a)**(1./3.))
    Z2 = (3.*a2 + Z1*Z1)**(0.5)
    r = 3. + Z2 - np.sign(a)*((3.-Z1) * (3.+Z1+2.*Z2))**(0.5)
    return r

def ISCO_energy(a):
    """ISCO Energy
    Phys.Rev.D 62, 124022 Ori&Thorne
    eq(2.5), (2.6)

    Args:
        a (float or numpy.ndarray): Spin parameter of black hole

    Returns:
        Eisco (float or numpy.ndarray): ISCO energy
    """

    a2 = a*a
    Z1 = 1. + (1-a2)**(1./3.) * ((1+a)**(1./3.) + (1-a)**(1./3.))
    Z2 = (3.*a2 + Z1*Z1)**(0.5)
    r = 3. + Z2 - np.sign(a)*((3.-Z1) * (3.+Z1+2.*Z2))**(0.5)

    numerator = 1. - 2./r + a/(r**1.5)
    denominator = (1. - 3./r + 2.*a/(r**1.5))**0.5
    Eisco = numerator / denominator
    return Eisco


def ISCO_angular_momentum(a):
    """ISCO angular momentum
    Phys.Rev.D 62, 124022 Ori&Thorne
    eq(2.8)

    Args:
        a (float or numpy.ndarray): Spin parameter of black hole

    Returns:
        Jisco (float or numpy.ndarray): ISCO angular momentum
    """

    a2 = a*a
    Z1 = 1. + (1-a2)**(1./3.) * ((1+a)**(1./3.) + (1-a)**(1./3.))
    Z2 = (3.*a2 + Z1*Z1)**(0.5)
    r = 3. + Z2 - np.sign(a)*((3.-Z1) * (3.+Z1+2.*Z2))**(0.5)

    numerator = 2.0 * (3.0 * r**0.5 - 2.0*a)
    denominator = (3.0 * r)**0.5
    Jisco = numerator / denominator
    return Jisco

def remnantmass(m1, m2, a1, a2, arem=None):
    """Remnant mass of binary black hole merger
    arXiv: 1610.09713 Healy&Lousto, eq(1)

    Args:
        m1 (float or numpy.ndarray): primary mass of BH
        m2 (float or numpy.ndarray): secondary mass of BH
        a1 (float or numpy.ndarray): spin parameter of primary BH
        a2 (float or numpy.ndarray): spin parameter of secondary BH

    Returns:
        Mrem (float or numpy.ndarray): remnant mass
    """

    M0 = 0.951659
    K1 = -0.051130
    K2a = -0.005699
    K2b = -0.058064
    K2c = -0.001867
    K2d = 1.995705
    K3a = 0.004991
    K3b = -0.009238
    K3c = -0.120577
    K3d = 0.016417
    K4a = -0.060721
    K4b = -0.001798
    K4c = 0.000654
    K4d = -0.156626
    K4e = 0.010303
    K4f = 2.978729
    K4g = 0.007904
    K4h = 0.000631
    K4i = 0.084478

    m = m1+m2
    eta = m1*m2/(m**2.0)
    q = m1/m2
    S1 = a1*m1*m1
    S2 = a2*m2*m2

    dm = (m1-m2)/m
    dm2 = dm*dm
    dm3 = dm2*dm
    dm4 = dm3*dm
    dm6 = dm4*dm2
    S = (S1+S2)/m/m
    S2 = S*S
    S3 = S2*S
    S4 = S3*S
    D = (S2/m2 - S1/m1)/m
    D2 = D*D
    D3 = D2*D
    D4 = D3*D

    if arem is None:
        Eisco = ISCO_energy(S/(m**2.0))
    else:
        Eisco = ISCO_energy(arem)

    P1 = M0 + K1*S + K2a*D*dm + K2b*S2
    P2 = K2c*D2 + K2d*dm2 + K3a*D*S*dm
    P3 = K3b*S*D2 + K3c*S3
    P4 = K3d*S*dm2 + K4a*D*S2*dm
    P5 = K4b*D3*dm + K4c*D4 + K4d*S4
    P6 = K4e*D2*S2 + K4f*dm4 + K4g*D*dm3
    P7 = K4h*D2*dm2 + K4i*S2*dm2
    R = (1. + eta*(Eisco+11.))*dm6

    Mrem = m * ( ((4.*eta)**2.) * (P1+P2+P3+P4+P5+P6+P7) + R )

    return Mrem



def remnantspin(m1, m2, a1, a2, eps=1e-6, da=1e-6):
    """Remnant spin
    arXiv: 1610.09713 Healy&Lousto, eq(2)

    Args:
        m1 (float or numpy.ndarray): primary mass of BH
        m2 (float or numpy.ndarray): secondary mass of BH
        a1 (float or numpy.ndarray): spin parameter of primary BH
        a2 (float or numpy.ndarray): spin parameter of secondary BH

    Returns:
        arem (float or numpy.ndarray): spin parameter of remnant BH
    """

    L0 = 0.686732
    L1 = 0.613285
    L2a = -0.148530
    L2b = -0.113826
    L2c = -0.003240
    L2d = 0.798011
    L3a = -0.068782
    L3b = 0.001291
    L3c = -0.078014
    L3d = 1.557286
    L4a = -0.005710
    L4b = 0.005920
    L4c = -0.001706
    L4d = -0.058882
    L4e = -0.010187
    L4f = 0.964445
    L4g = -0.110885
    L4h = -0.006821
    L4i = -0.081648

    m = m1+m2
    eta = m1*m2/(m**2.0)
    q = m1/m2
    S1 = a1*m1*m1
    S2 = a2*m2*m2

    dm = (m1-m2)/m
    dm2 = dm*dm
    dm3 = dm2*dm
    dm4 = dm3*dm
    dm6 = dm4*dm2
    S = (S1+S2)/m/m
    S2 = S*S
    S3 = S2*S
    S4 = S3*S
    D = (S2/m2 - S1/m1)/m
    D2 = D*D
    D3 = D2*D
    D4 = D3*D


    P1 = L0 + L1*S
    P2 = L2a*D*dm + L2b*S2 + L2c*D2 + L2d*dm2
    P3 = L3a*D*S*dm + L3b*S*D2 + L3c*S3
    P4 = L3d*S*dm2 + L4a*D*S2*dm + L4b*D3*dm
    P5 = L4c*D4 + L4d*S4 + L4e*D2*S2
    P6 = L4f*dm4 + L4g*D*dm3
    P7 = L4h*D2*dm2 + L4i*S2*dm2

    def func(arem):
        Jisco = ISCO_angular_momentum(arem)
        R = S*(1.0+8.0*eta)*dm4 + eta*Jisco*dm6
        arem = ((4.*eta)**2.) * (P1+P2+P3+P4+P5+P6+P7) + R
        return arem

    arem_trial = S / (m**2.)
    count = 0
    flist = []
    dflist = []
    while True:
        fa = func(arem_trial)
        dfda = (func(arem_trial + da) - fa) / da
        flist.append(fa)
        dflist.append(dfda)
        arem_new = arem_trial - (fa - arem_trial) / (dfda - 1)
        if np.abs(arem_new - arem_trial) < eps:
            break
        else:
            arem_trial = arem_new
        count+=1
        if count>100:
            break
    return arem_new
