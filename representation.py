import math

import matplotlib.cm as cm
import matplotlib
import numpy as np
from qutip import *
np.set_printoptions(linewidth=350)

b3 = Bloch3d()
def spinor(coefficients):
    """
    Constructs a spinor of spin (coefficients.size-1)/2 with given coefficients
    :param coefficients: coefficients of spinor from top to bottom
    :return: spinor as quantum object
    """
    return Qobj(coefficients)
    """
    Return the stereographic mapping for zeta as defined by Barnett
    :param theta: theta in Bloch sphere representation
    :param phi: phi in Bloch sphere representation
    :return: e^iphi tan(theta/2)
    """
    return np.e**(1.0j * phi) * np.tan(theta/2)


def projection_onto_spin_half(spinor): # todo add handling for m = 0
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

#TODO add down states to drawing - check if number of roots < degree: then have 0


def projections_into_states(thetas, phis):
    """
    :param thetas:
    :param phis:
    :return: 3xN matrix: first two rows represent spin half states and last represents number of occurrences
    """
    # todo check if eg (1, 1, 0, 0, 0) and with more 0s isn't broken - think it's fine as coefficients move around..
    n = thetas.size
    newstates = np.zeros([3, thetas.size], dtype=complex)
    newstates[:2, :] = -1
    for i in range(n): # fix 0 root being mapped to (0, 0) and therefore not plotting. probably similar with south pole TODO
        state = np.array([np.e**(1.0j * phis[i]/2)*np.cos(thetas[i]/2),
                             np.e**(-1.0j * phis[i]/2)*np.sin(thetas[i]/2)]).round(10) # is this 0 for spin1 but 1 for everything higer??
        if state in newstates[0:2, :].T:
            location = np.argmax(newstates[0:2, :] == state)
        else:
            location = np.argmax(newstates[0, :]==-1)
            newstates[0:2, location] = state
        newstates[2, location] = newstates[2, location] + 1
    newstates = newstates[:, ~np.all(newstates[0:1]==-1, axis=0)]
    #for i in range(newstates.shape[1]):
    #    if np.allclose(newstates[0:1, i], np.array([0, 0])):
    #       newstates[0, i] = 1
    return newstates


def decompose_spinor(spinor, gauge=1):
    _, thetas, phis = projection_onto_spin_half(spinor)
    states = projections_into_states(thetas, phis)
    return states


def draw_spinor_projection(spinor, gauge=1, d3=True, clear=True, return_states=False, kind='vector', show=True):
    states = decompose_spinor(spinor)
    objects = [Qobj(states[:2, i]) for i in range(states.shape[1])]
    if d3:
        bloch = b3
    else:
        bloch = b
    if clear:
        bloch.clear()
    for i, number in enumerate(states[2, :]):
        if number > 1:
            bloch.add_annotation(objects[i], "x%d" %int(number))
    bloch.add_states(objects, kind)
    if show:
        bloch.show()
    if return_states:
        return objects

# todo design tests
# todo calculate Berry phase
# todo implement a function which draws lines (use point parameter method='l')
# todo draw animation of moving spinors around a curve (eg phi = 0)
# todo draw animation of moving spinors around arbitrary curve


def construct_polar_states(states):
    n = len(states)
    thetas = np.zeros(n, dtype=complex)
    phis = np.zeros(n, dtype=complex)
    for k in range(n):
        thetas[k] = 2*np.arccos(states[k][0])
        if thetas[k] != 0.0 and thetas[k] != np.pi:
            phis[k] = np.log(states[k][1]/np.sin(thetas[k]/2))/1.0j
        else:
            phis[k] = 0
    thetas = thetas.real # cast to real to avoid numerical computation related issues later
    phis = phis.real
    states0 = projections_into_states(thetas, phis)
    for i in range(len(states0)): #  todo check if this is correct - this changes sign to fix something later
        if states[i] != states0[i]:
            thetas[i] = -thetas[i]
            print("flipped sign for $\\theta=%.3f$" % thetas[i])
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
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return x, y, z


def point_to_polar(v):
    theta = np.arccos(v[2, :])
    cos_phi = v[0, :]/np.sin(theta)
    return theta, np.arccos(cos_phi)


def draw_points(spinor=None, thetas=None, phis=None, d3=True, clear=True, method='s', return_points=False, show=True):
    if thetas is None or phis is None:
        _, thetas, phis = projection_onto_spin_half(spinor) # todo fix none spinor thing
    states = projections_into_states(thetas, phis)[:2]
    x = [expect(sigmax(), Qobj(s)) for s in states.T]
    y = [expect(sigmay(), Qobj(s)) for s in states.T]
    z = [expect(sigmaz(), Qobj(s)) for s in states.T]
    if d3:
        bloch = b3
    else:
        bloch = b
    if clear:
        bloch.clear()
    bloch.add_points([x, y, z])
    if show:
        bloch.show()
    if return_points:
        return x, y, z


def refresh_N(eta):
    return Qobj(np.array([np.sin(eta) / np.sqrt(2), 0, np.cos(eta), 0, np.sin(eta) / np.sqrt(2)]))

#todo fix bloch3d plotting normals

eta = np.pi/4
F = Qobj(np.array([1, 0, 0, 0, 0]))
N = Qobj(np.array([np.sin(eta)/np.sqrt(2), 0, np.cos(eta), 0, np.sin(eta)/np.sqrt(2)]))
T = Qobj(np.array([np.sqrt(1/3), 0, 0, np.sqrt(2/3), 0]))
b = Bloch()
#b3 = Bloch3d()


def comb(n, k):
    return math.factorial(n)/(math.factorial(k) * math.factorial(n - k))
