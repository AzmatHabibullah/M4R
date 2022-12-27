import numpy as np
from qutip import *
from representation import draw_spinor_projection, draw_points, projection_onto_spin_half, b, b3
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_from_e1(vec, alpha, N=50):
    e1 = np.array([1, 0, 0])
    v = np.cross(e1, vec)
    c = np.dot(e1, vec)
    R = np.eye(3) + skew(v) + skew(v).dot(skew(v)) / (1 + c)
    xp = [np.cos(alpha) for t in np.linspace(0, 2 * np.pi, N)]
    yp = [np.sin(alpha) * np.cos(t) for t in np.linspace(0, 2 * np.pi, N)]
    zp = [np.sin(alpha) * np.sin(t) for t in np.linspace(0, 2 * np.pi, N)]
    return R.dot([xp, yp, zp])


"""def plot_angles():
    v = np.random.rand(3)
    v = v / np.linalg.norm(v)
    b3.clear()
    b3.add_points(v)
    b3.add_points(rotation_from_e1(v, np.pi / 2))
    b3.show()
"""


def circular_sphere(a, b, g, precision=100, cartesian=True):
    t = np.linspace(0, 2 * np.pi, precision)
    x = np.sin(a) * np.cos(b) * np.cos(g) * np.cos(t) + \
        np.sin(a) * np.sin(g) * np.sin(t) - np.cos(a) * np.sin(b) * np.cos(g)
    y = -np.sin(a) * np.cos(b) * np.sin(g) * np.cos(t) + \
        np.sin(a) * np.cos(g) * np.sin(t) + np.cos(a) * np.sin(b) * np.sin(g)
    z = np.sin(a) * np.sin(b) * np.cos(t) + np.cos(a) * np.cos(b)
    if cartesian:
        return np.array([x, y, z])
    theta = np.arccos(z / (np.sqrt(x ** 2 + y ** 2 + z ** 2)))
    phi = np.arctan(y / x)
    return np.array([theta, phi])


def plot_evolution(spinors, d3=False, method='s', clear=True, return_psi=False, plot_p1=False):
    """

    :param spinors: np array of spinors N spinors given in each column
    :param d3: 3d or not
    :param method:
    """
    points, dim = spinors.shape
    F = (dim - 1) / 2
    cmap = cm.get_cmap('viridis', points)
    if d3:
        bloch = b3
        bloch.point_size = 0.06
    else:
        bloch = b
        bloch.point_size = [45]  # for some reason this takes a list
    bloch.point_color = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    bloch.point_marker = ['o']
    theta_evolution = np.zeros([points, int(2 * F)])
    phi_evolution = np.zeros([points, int(2 * F)])
    for i, spinor in enumerate(spinors):
        _, thetas, phis = projection_onto_spin_half(Qobj(spinor))
        print(thetas)
        print(phis)
        theta_evolution[i] = thetas
        phi_evolution[i] = phis

        draw_points(thetas=thetas, phis=phis, d3=d3, clear=False,
                    show=False, method=method)
    bloch.show()
    if plot_p1:
        # print(theta_evolution)
        # print(phi_evolution)
        for i in range(int(2 * F)):
            plt.plot(np.linspace(1, points, points), theta_evolution[:, i], label="$\\theta_%d$" % (i + 1))
            plt.plot(np.linspace(1, points, points), phi_evolution[:, i], label="$\\phi_%d$" % (i + 1))
        # todo change to show t on the x axis
        # todo fix theta and phi tracking
        plt.legend()
        plt.grid()
        plt.show()
    if clear:
        bloch.clear()
    if return_psi:
        return theta_evolution, phi_evolution
    # todo fix add to points to work with the new state things


def solve_system(initial_value, t0=0, t1=1, precision=100, draw_m=False, d3=False, func=None):
    if func is None:
        func = c_nought_zero
    alpha = np.pi / 3
    f = 1 + np.sin(alpha) / (np.sqrt(2) * np.cos(alpha))
    psi0 = Qobj(initial_value).unit()
    F = (initial_value.size - 1) / 2  # todo fix
    m = Qobj(np.array([expect(jmat(F, "x"), psi0), expect(jmat(F, "y"), psi0),
                       expect(jmat(F, "z"), psi0)]))
    b.vector_color = ["b", "b", "c", "c", "y", "y"]

    if draw_m and np.linalg.norm(m) != 0.0:
        draw_spinor_projection(m, d3=d3, clear=False, show=False)
        print("m\n ", m)
        print("magnitude of m is", np.linalg.norm(m))

    def zfunc(psi, t):
        return func(psi, m)

    t = np.linspace(t0, t1, precision)

    psi, infodict = odeintz(zfunc, initial_value / np.linalg.norm(initial_value), t, full_output=True)
    return psi, infodict


def odeintz(func, z0, t, **kwargs):
    from scipy.integrate import odeint
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    """_unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))"""

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z


def plot(ic, precision=50, t1=1, draw_m=True, return_psi=False, d3=False, func=None, plot_p1=False, clear=True):
    if func is None:
        func = c_nought_zero
    psi, infodict = solve_system(ic, t1=t1, precision=precision, draw_m=draw_m, func=func)
    draw_spinor_projection(Qobj(ic), d3=d3, clear=False, show=False)
    draw_spinor_projection(Qobj(psi[-1]), d3=d3, clear=False, show=False)
    plot_evolution(psi, clear=clear, plot_p1=plot_p1, d3=d3)
    if return_psi:
        return psi


def plot_zero_m(precision=100, tol=10e-5, F=1):
    vec = np.random.normal(size=(precision, 3))
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    xp = vec[:, 0]
    yp = vec[:, 1]
    zp = vec[:, 2]
    b.clear()
    for x in xp:
        for y in yp:
            for z in zp:
                psi = Qobj(np.array([x, y, z]))
                m = Qobj(np.array([expect(jmat(F, "x"), psi), expect(jmat(F, "y"), psi),
                                   expect(jmat(F, "z"), psi)]))
                if np.linalg.norm(m) < tol:
                    print([x, y, z])
                    b.add_points([x, y, z])
    b.show()


def mean_field_RHS(F=1, c0=1, c2=1):
    def to_return(psi, m):
        p = Qobj(psi)
        m = Qobj(np.array([expect(jmat(F, "x"), p), expect(jmat(F, "y"), p),
                           expect(jmat(F, "z"), p)]))
        return c2 * (m[0, 0] * jmat(F, "x") + m[1, 0] * jmat(F, "y") + m[2, 0] * jmat(F,
                                                                                      "z")) * psi * 1 / 1.0j + c0 * 1 / 1.0j * psi  # *np.abs(psi**2)

    return to_return


def spin_2_RHS(F=2, c0=1, c1=1, c2=1):
    def to_return(psi, m):
        p = Qobj(psi)
        A = (2 * psi[0] * psi[-1] - 2 * psi[1] * psi[-2] + psi[3] ** 2) / np.sqrt(5)
        rhs = c0 * psi
        fz = expect(jmat(F, "z"), p)
        fplus = expect(jmat(F, "+"), p)
        fminus = np.conjugate(fplus)
        rhs[2] += np.sqrt(6) / 2 * c1 * (fplus * psi[1] + fminus * psi[-2]) \
                  + c2 / np.sqrt(5) * A * np.conjugate(psi[2])
        rhs[0] += 2 * c1 * fz * psi[0] + c1 * fminus * psi[1] + c2/np.sqrt(5)*A*np.conjugate(psi[-1])
        rhs[-1] += -2*c1 * fz * psi[-1] + c1 * fplus*psi[-2] + c2/np.sqrt(5)*A*np.conjugate(psi[0])
        rhs[1] += c1*fz*psi[1] + c1*(np.sqrt(6)/2 * fminus * psi[2] + fplus * psi[0]) - \
                  c2/np.sqrt(5)*A*np.conjugate(psi[-2])
        rhs[-2] += -c1*fz*psi[-2] + c1*(np.sqrt(6)/2*fplus*psi[2] + fminus * psi[-1]) - \
                   c2/np.sqrt(5)*A*np.conjugate(psi[1])
        return rhs
    return to_return
