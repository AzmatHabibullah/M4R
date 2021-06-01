from qutip import *
from representation import decompose_spinor
from mean_field_dynamics import plot, mean_field_RHS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

b = Bloch()
b3 = Bloch3d()

def chapter_one():
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

        b.clear()
        f = 2
        n = 2*f + 1
        b.add_states(decompose_spinor(basis(n, 0) + basis(n, n-1)))
        b.show()
        b.save(name=name+"noon-state-"+n+".png")

        b3.clear()
        f = 33
        n = 2 * f + 1
        b3.add_states(decompose_spinor(Qobj(np.array([1]*n))))
        b3.show()
# todo fix colouring for m, etc

def chapter_four():
    c2_params = [1]
    c0_params = [0,0,0]
    #psi_params = [np.array([1, 2, 0]), np.array([3, 1, 16j-4j]), np.array([4j, 12+1j, 3-3j])]
    psi_params = [np.array([1, 3, 1, 3, 1])]
    for psi, c0, c2 in zip(psi_params, c0_params, c2_params):
        F = (psi.size-1)/2
        plt.title("Evolution of $\psi=%s$ for $F=%.1f, c_0=%d, c_2=%d$" % (np.array2string(psi), F, c0, c2))
        plot(t1=2, return_psi=True, precision=100, d3=True,
             ic = psi, func=mean_field_RHS(F=F, c0=c0, c2=c2), plot_p1=True)

chapter_four()






