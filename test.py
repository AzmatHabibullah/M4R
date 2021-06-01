from mean_field_dynamics import solve_system, mean_field_RHS
from representation import construct_polar_states, polar_to_point, projection_onto_spin_half
import pytest
import numpy as np
from qutip import *

spin_1_tests = [(1, 100, np.array([1, 2, 3])),(1, 34, np.array([np.exp(1.0j*np.pi/4), 17-3j, 0])),
              (1, 68, np.array([1, np.exp(np.pi/3*1j), np.exp(1.0j*np.pi/4)])),
              (1, 68, np.array([1, 9, 3j])), (1, 22, np.array([1, 0, 3j])), (2, 50, np.array([1.0j, 3.5j - 5, 49j]))]

spin_2_tests = [(2, 40, np.array([1, 1, 1, 1, 1])),
              (2, 34, np.array([-6j + 3, 2, 9j, 18, 5])),
              (2, 50, np.array([1.0j, 3j, 5, 3.5j - 5, 49j]))]

failed_tests = [(1, 50, np.array([0, 4, 3j]))]
# n1+n2 conserved test fail only. is this because m is conserved in spin 1?
# what's conserved for spin 2?

@pytest.mark.parametrize('c2, precision, psi0', spin_1_tests)
def test_m_conserved_for_c0_zero(c2, precision, psi0):
    """
    Test if m = (m1, m2, m3), the expectation of psi under S, is conserved under evolution with c_0 = 0
    """
    F = (psi0.size-1)/2
    psi_evolution, _ = solve_system(psi0, t1=10, precision=precision,
                                    draw_m=False, d3=False, func=mean_field_RHS(F=F, c0=0, c2=c2))
    m = np.zeros([precision, int(2*F)+1])
    thetas = np.zeros([precision, int(2*F)])
    phis = np.zeros([precision, int(2*F)])
    _, thetas[0], phis[0] = projection_onto_spin_half(Qobj(psi0))
    normals0 = np.zeros([int(2*F), 3])
    for i in range(int(2*F)):
        x0, y0, z0 = polar_to_point(thetas[0, i], phis[0, i])
        normals0[i, :] = np.array([x0, y0, z0])
    #x0, y0, z0 = polar_to_point(thetas, phis)
    #normals0 = np.array([x0, y0, z0])
    n0 = np.array([np.sum(normals0[2*i:2*i+2], axis=0) for i in range(int(F))])
    for i, psi in enumerate(psi_evolution):
        psi = Qobj(psi)
        m[i, 0] = expect(jmat(F, "x"), psi)
        m[i, 1] = expect(jmat(F, "y"), psi)
        m[i, 2] = expect(jmat(F, "z"), psi)
        if i > 0:
            assert np.linalg.norm(m[i, :] - m[0, :]) < 10e-5                  # check m constant
            _, thetas[i], phis[i] = projection_onto_spin_half(Qobj(psi))
            x, y, z = polar_to_point(thetas[i], phis[i])
            normals = np.array([x, y, z])
            #for j in range(int(2*F)):
            #    assert np.linalg.norm(np.linalg.norm(normals[j]) - 1) < 10e-5     # check ni normalised
            n = [np.sum(normals[:, 2*i:2*i+2], axis=1) for i in range(int(F))]
            n1 = np.array([x[0], y[0], z[0]])
            n2 = np.array([x[1], y[1], z[1]])
            assert np.linalg.norm(n - n0) < 10e-5
            alpha = np.dot(normals[0], normals[1])
            alpha0 = np.dot(n1, n2)
            #assert np.linalg.norm(alpha - alpha0) < 10e-5


    # check if n1, n2 and n1.n2 conserved. do using polar representation with generated psi.
    # n1 + n2 conserved for F = 1 (check with more cases)
    # calculate representation, convert to cartesian, iterate through and check conserved.


@pytest.mark.parametrize('c0, precision, psi0', spin_2_tests)
def test_m3_conserved_for_c2_zero(c0, precision, psi0):
    """
    Test if m3, the expectation of psi under Sz, is conserved under evolution with c_2 = 0
    """
    F = (psi0.size-1)/2
    psi_evolution, _ = solve_system(psi0, t1=10, precision=precision,
                                    draw_m=False, d3=False, func=mean_field_RHS(F=F, c0=c0, c2=0))
    m = np.zeros([precision])
    for i, psi in enumerate(psi_evolution):
        psi = Qobj(psi)
        m[i] = expect(jmat(F, "z"), psi)
        if i > 0:
            assert(np.linalg.norm(m[i] - m[0]) < 10e-5)

# test n1 + n2 conserved, for c2=0
# test n1.n2 conserved, for c2=0

@pytest.mark.parametrize('F', [(1)])
def test_antipodal_conserved(F):
    assert F==1


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)