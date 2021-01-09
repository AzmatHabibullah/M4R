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


def projection_onto_spin_half(spinor):
    spinor = spinor.unit() # normalise
    F = (spinor.shape[0]-1)/2
    poly_coeffs = np.zeros(spinor.shape[0], dtype=complex)
    for k in range(spinor.shape[0]):
        poly_coeffs[k] = np.sqrt(comb(int(2*F), k))*spinor.full()[k, 0].conj()
    # highest coefficient first
    roots = np.roots(poly_coeffs)
    thetas = 2*np.arctan(np.abs(roots))
    phis = np.angle(roots)
    return poly_coeffs, thetas, phis


def projections_into_states(thetas, phis, gauge=1):
    if gauge==0:
        states = [Qobj(np.array([np.e**(1.0j * phis[i]/2)*np.cos(thetas[i]/2),
                             np.e**(-1.0j * phis[i]/2)*np.sin(thetas[i]/2)]).round(10)) for i in range(thetas.size)]
    if gauge==1:
        states = [Qobj(np.array([np.cos(thetas[i] / 2),
                                 np.e ** (1.0j * phis[i]) * np.sin(thetas[i] / 2)]).round(10)) for i in range(thetas.size)]

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

# todo calculate Berry phase
# todo fix branch cut issues for inversion
# todo implement a function which draws lines (use point parameter method='l')
# todo draw animation of moving spinors around a curve (eg phi = 0)
# todo draw animation of moving spinors around arbitrary curve


def construct_polar_states(states, gauge=1):
    n = len(states)
    F = n/2
    thetas = np.zeros(n, dtype=complex)
    phis = np.zeros(n, dtype=complex)
    for k in range(n):
        if gauge ==1:
            thetas[k] = 2*np.arccos(states[k][0][0])
            if thetas[k] != 0.0 and thetas[k] != np.pi:
                phis[k] = np.log(states[k][1][0]/np.sin(thetas[k]/2))/1.0j
            else:
                phis[k] = 0
    thetas = thetas.real
    phis = phis.real
    states0 = projections_into_states(thetas, phis, gauge)
    for i in range(len(states0)):
        if states[i] != states0[i]:
            thetas[i] = -thetas[i]
    return thetas, phis


def build_spinor_from_polars(thetas, phis):
    n = thetas.size
    F = n/2
    roots = [np.tan(thetas[i]/2) * np.e ** (1.0j * phis[i]) for i in range(n)]
    poly_coeffs = np.polynomial.polynomial.polyfromroots(roots)
    spinor_coeffs = np.zeros(int(2*F + 1), dtype=complex)
    for k in range(int(2*F + 1)):
        spinor_coeffs[k] = poly_coeffs[k]/np.sqrt(comb(int(2*F), k)).conj()
    unnormalised_spinor = Qobj(spinor_coeffs)
    return unnormalised_spinor/unnormalised_spinor.norm()


def reconstruct_spinor(states, gauge=1):
    thetas, phis = construct_polar_states(states, gauge)
    return build_spinor_from_polars(thetas, phis)


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
        b.add_points([x, y, z], method)
        b.show()
    if return_points:
        return x, y, z


def refresh_N(eta):
    return Qobj(np.array([np.sin(eta) / np.sqrt(2), 0, np.cos(eta), 0, np.sin(eta) / np.sqrt(2)]))

def normal(spinor):
    return np.array([expect(sigmax(), spinor), expect(sigmay(), spinor), expect(sigmaz(), spinor)])
    # this is just the points!

def normals(spinor):
    states = decompose_spinor(spinor)
    return [normal(states[i]) for i in range(len(states))]

#todo fix bloch3d plotting normals

eta = np.pi/4
F = Qobj(np.array([1, 0, 0, 0, 0]))
N = Qobj(np.array([np.sin(eta)/np.sqrt(2), 0, np.cos(eta), 0, np.sin(eta)/np.sqrt(2)]))
T = Qobj(np.array([np.sqrt(1/3), 0, 0, np.sqrt(2/3), 0]))
b = Bloch()
b3 = Bloch3d()
