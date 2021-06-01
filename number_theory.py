import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def plot(n):
    for p in range(n):
        coefficients = np.array([np.sqrt(sp.binom(p, k)) for k in range(p+1)])
        roots = np.roots(coefficients)
        plt.plot(roots.real, roots.imag, '.')
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), 'black')
    plt.show()
