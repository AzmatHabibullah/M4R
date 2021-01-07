# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from qutip import *
from qutip import piqs
import numpy as np

omega = 0.5
theta = np.pi/2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    piqs.spin_algebra()
    berry_curvature()


def cos_wt(t, args):
    return np.cos(omega*t)


def sin_wt(t, args):
    return np.sin(omega*t)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


def t_dep_H():
    size = 2
    a = qutip.destroy(size)
    ad = qutip.create(size)
    n = qutip.num(size)
    I = qutip.qeye(size)
    function_form = QobjEvo([np.cos(theta) * sigmaz(), [np.sin(theta) * sigmax(), cos_wt], [np.sin(theta) * sigmay(), sin_wt]])
    function_f = QobjEvo([np.cos(theta) * I, [np.sin(theta) * sigmax(), cos_wt], [np.sin(theta) * sigmay(), sin_wt]])
    function_form(2).eigenstates(2)
    berry_curvature()

def construct_H():
    mesh_size = 50
    theta_params = np.linspace(0, np.pi, mesh_size)
    phi_parms = np.linspace(0, np.pi * 2, mesh_size)
