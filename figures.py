from qutip import *
from representation import draw_spinor_projection, projection_onto_spin_half, polar_to_point, point_to_polar, b,b3
from mean_field_dynamics import plot, mean_field_RHS, circular_sphere, spin_2_RHS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plot(t1=1, return_psi=True, precision=100, d3=False,
     ic=np.array([1, 1, 1j, 1, 1]), func=spin_2_RHS(F=2, c0=1, c1=0, c2=0), plot_p1=False)
def chapter_two():
    header = "plots\\"
    def section_one(folder="ch2"):
        name = header + folder + "\\"
        b.clear()
        b.add_states(basis(2, 0))
        b.add_states(basis(2, 1))
        b.show()
        b.save(name=name+"basis-spinors.png")

        b.clear()
        b.add_states(Qobj(np.array([1, 2])/np.sqrt(2)))
        b.add_states(Qobj(np.array([3, 4+5*1.0j]) / np.sqrt(3**2 + 4**2 + 5**2)))
        b.show()
        b.save(name=name+"other-states.png")

    def section_two(folder="ch2\\sec2"):
        name = header+folder+"\\"
        b.clear()
        draw_spinor_projection(Qobj(np.array([1, 0, 0, 0])), d3=False)
        b.show()
        b.save(name=name+"repeated-north-pole.png")

        b.clear()
        draw_spinor_projection(Qobj(np.array([3, 0, 0, 0, 0, 0, 5j])), d3=False)
        b.show()
        b.save(name=name+"hexagon.png")

        b.clear()
        # phi = pi, theta = pi/2
        draw_spinor_projection(Qobj(np.array([-np.exp(-2j*np.pi), -np.exp(-1j*np.pi)/np.tan(np.pi/2)/np.sqrt(2), 1])), d3=False)
        b.show()
        b.save(name=name+"(pi-over-2, pi) antipodal.png")

        b.clear()
        # phi = pi/3, theta = pi/4
        draw_spinor_projection(Qobj(np.array([np.exp(-2j*np.pi/3)*np.tan(np.pi/4/2)**2,
                                              -np.sqrt(2)*np.exp(-1j*np.pi/3)*np.tan(np.pi/4/2), 1])), d3=False)
        b.show()
        b.save(name=name+"(pi-over-4, pi-over-3) rpt")  # todo why isn't this showing x2?

    section_two()

# todo fix colouring for m, etc

def helper():
    c2_params = [1]
    c0_params = [0, 0, 0]
    # psi_params = [np.array([1, 2, 0]), np.array([3, 1, 16j-4j]), np.array([4j, 12+1j, 3-3j])]
    psi_params = [np.array([-(-1 - 1 + np.sqrt(3)), -(-1 + np.sqrt(3)), 1])]
    for psi, c0, c2 in zip(psi_params, c0_params, c2_params):
        F = (psi.size - 1) / 2
        plt.title("Evolution of $\psi=%s$ for $F=%.1f, c_0=%d, c_2=%d$" % (np.array2string(psi), F, c0, c2))
        plot(t1=1, return_psi=True, precision=100, d3=True,
             ic=psi, func=mean_field_RHS(F=F, c0=c0, c2=c2), plot_p1=True)


def chapter_four():
    header = "plots\\"
    def section_three_c0_0(folder="ch4\\sec3"):
        name = header + folder + "\\"
        b.clear()
        # (-m1 + im2, m3, m1+im2)
        plot(np.array([-3+3j, 4, 3+3j]),
             draw_m=False, func=mean_field_RHS(1, 0, 1), clear=False)
        b.save(name=name+"zero-energy-state")
                # (1, -(m3 + sqrt(m3^2 + 2(m1^2 + m2^2))/(m1-im2)),
                # 1/(m1 - im2)^2 * ((m1 + im2)^2 + m3(m3 + sqrt(...)))

        b.clear()
        plot(np.array([1, -1-np.sqrt(3), 1 + 1 + np.sqrt(3)]),
             draw_m=False, func=mean_field_RHS(1, 0, 1), clear=False)
        b.save(name=name+"non-zero-energy-state")


        b.clear()
        plot(np.array([5, 5 + 5j, 1]),
             draw_m=False, func=mean_field_RHS(1, 0, 1), clear=False, t1=4)
        b.save(name=name+"great-circle-1")

        b.clear()
        plot(np.array([3 + 3j, -2 + 1j, 2 + 1j]),
             draw_m=False, func=mean_field_RHS(1, 0, 1), clear=False, t1=5)
        b.save(name=name+"great-circle-2")

    def section_three_c2_0(folder="ch4\\sec3\\c2_0"):
        name = header + folder + "\\"
        b.clear()
        # (-m1 + im2, m3, m1+im2)
        plot(np.array([1, 1, 1]),
             draw_m=False, func=mean_field_RHS(1, 1, 1), clear=False)
        #b.save(name=name + "zero-energy-state")

    section_three_c2_0()

    def spin_2(folder="ch4\\spin2"):
        name = header + folder + "\\"

        b.clear()
        plot(np.array([1, 1, 1, 1, 1]),
             draw_m=False, func=spin_2_RHS(2, 0, 0, 1), clear=False, d3=False, t1=1)
        b.save(name=name + "psi-5-ones")

        b.clear()
        plot(np.array([1, 0, 0, 0, 1]),
             draw_m=False, func=spin_2_RHS(2, 0, 0, 1), clear=False, d3=False, t1=1)
        b.save(name=name + "psi-e1ande5")

    spin_2()

def draw_with_sphere(ic, d3=False, plot_p1=False, t1=1):
    if d3:
        bloch = b3
    else:
        bloch = b

    _, thetas, phis = projection_onto_spin_half(Qobj(ic))
    x0, y0, z0 = polar_to_point(thetas, phis)
    normals = np.array([x0, y0, z0])
    n1 = normals[:, 0]
    n2 = normals[:, 1]
    angle = np.arccos(n1.dot(n2))
    midpoint = n1 + n2
    midpoint = midpoint / np.linalg.norm(midpoint)
    bloch.add_vectors(midpoint)
    print(midpoint)
    beta, gamma = point_to_polar(midpoint)
    print(beta, gamma)
    x, y, z = circular_sphere(angle / 2, thetas.mean(), phis.mean(), precision=50)
    bloch.add_points([x, y, z])
    psis = plot(t1=t1, return_psi=True, precision=50, d3=d3, ic=ic, func=mean_field_RHS(F=1, c0=0, c2=1),
        plot_p1 = plot_p1)

chapter_four()