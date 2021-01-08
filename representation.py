import numpy as np
from qutip import *
from math import comb
from cmath import polar

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


def maximally_polarised_state(F):
    coefficients = np.zeros(2*F)


def compute_coefficient(alpha):
    return 2

def projection_onto_spin_half(spinor):
    F = (spinor.shape[0]-1)/2
    poly_coeffs = np.zeros(spinor.shape[0], dtype=complex)
    for k in range(spinor.shape[0]):
        poly_coeffs[k] = np.sqrt(comb(int(2*F), k))*spinor.full()[k, 0].conj()
    roots = np.roots(poly_coeffs)
    thetas = np.abs(roots)
    phis = np.angle(roots)
    return poly_coeffs, thetas, phis


def projections_into_states(thetas, phis, gauge=1):
    if gauge==0:
        states = [Qobj(np.array([np.e**(1.0j * phis[i]/2)*np.cos(thetas[i]/2),
                             np.e**(-1.0j * phis[i]/2)*np.sin(thetas[i]/2)])) for i in range(thetas.size)]
    if gauge==1:
        states = [Qobj(np.array([np.cos(thetas[i] / 2),
                                 np.e ** (1.0j * phis[i]) * np.sin(thetas[i] / 2)])) for i in range(thetas.size)]
    return states


def decompose_spinor(spinor, gauge=1):
    _, thetas, phis = projection_onto_spin_half(spinor)
    states = projections_into_states(thetas, phis, gauge)
    return states


def draw_spinor_projection(spinor, gauge=1, d3=True, clear=True, return_states=False, kind='vector'):
    states = decompose_spinor(spinor, gauge)
    if d3:
        if clear:
            b3.clear()
        b3.add_states(states, kind)
        b3.show()
    else:
        if clear:
            b.clear()
        b.add_states(states, kind)
        b.show()
    if return_states:
        return states

# todo implement a function which draws lines (use point parameter method='l')
# todo draw animation of moving spinors around a curve (eg phi = 0)
# todo draw animation of moving spinors around arbitrary curve
# todo calculate Berry phase


def construct_polar_states(states, gauge=1):
    n = len(states)
    F = n/2
    thetas = np.zeros(n, dtype=complex)
    phis = np.zeros(n, dtype=complex)
    for k in range(n):
        if gauge ==1:
            thetas[k] = 2*np.arccos(states[k][0][0])
            if thetas[k] != 0:
                phis[k] = np.log(states[k][1][0]/np.sin(thetas[k]/2))/1.0j # todo fix branch cut issues
            else:
                phis[k] = 0
    thetas = thetas.real
    phis = phis.real
    return thetas, phis


def reconstruct_spinor(thetas, phis):
    n = thetas.size
    F = n/2
    roots = [thetas[i] * np.e ** (1.0j * phis[i]) for i in range(n)]
    poly_coeffs = np.polynomial.polynomial.polyfromroots(roots) # this is monic! see rescaling issues
    spinor_coeffs = np.zeros(int(2*F + 1), dtype=complex)
    for k in range(int(2*F + 1)):
        spinor_coeffs[k] = poly_coeffs[k]/np.sqrt(comb(int(2*F), k)).conj()
    return Qobj(spinor_coeffs)


def polar_to_point(thetas, phis):
    n = thetas.size
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return x, y, z


def draw_points(spinor, thetas=None, phis=None, d3=True, clear=True, method='s', return_points=False):
    if thetas is None or phis is None:
        _, thetas, phis = projection_onto_spin_half(spinor)
    x, y, z = polar_to_point(thetas, phis)
    if d3:
        if clear:
            b3.clear()
        b3.add_points([x, y, z], method)
        b3.show()
    else:
        if clear:
            b.clear()
        b3.add_points([x, y, z], method)
        b.show()
    if return_points:
        return x, y, z


def refresh_N(eta):
    return Qobj(np.array([np.sin(eta) / np.sqrt(2), 0, np.cos(eta), 0, np.sin(eta) / np.sqrt(2)]))


eta = np.pi/3
F = Qobj(np.array([1, 0, 0, 0, 0]))
N = Qobj(np.array([np.sin(eta)/np.sqrt(2), 0, np.cos(eta), 0, np.sin(eta)/np.sqrt(2)]))
T = Qobj(np.array([np.sqrt(1/3), 0, 0, np.sqrt(2/3), 0]))
b = Bloch()
b3 = Bloch3d()
