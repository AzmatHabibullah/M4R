import numpy as np
from qutip import Qobj

def spinor(coefficients):
    """
    Constructs a spinor of spin (coefficients.size-1)/2 with given coefficients
    :param coefficients: coefficients of spinor from top to bottom
    :return: spinor as quantum object
    """
    return Qobj(coefficients)


def zeta(theta, phi):
    """
    Return the stereographic mapping for zeta as defined by Barnett
    :param theta: theta in Bloch sphere representation
    :param phi: phi in Bloch sphere representation
    :return: e^iphi tan(theta/2)
    """
    return np.e**(1.0j * phi) * np.tan(theta/2)




def compute_coefficient(alpha):


def construct_polynomial(spinor):
