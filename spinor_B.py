#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 01:17:26 2019

@author: ssaumya7
"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *


N = 50
thetas = np.linspace(0, np.pi, N)
phis = np.linspace(0, 2*np.pi, N)
eigfs = np.zeros((N, N, 2, 2), dtype=complex)
for i in range(N):
    for j in range(N):
        # H = B.sigma, sigma is vector of Pauli matrices
        H = sigmax() * np.sin(thetas[i]) * np.cos(phis[j]) + sigmay() * np.sin(thetas[i]) * np.sin(phis[j]) + np.cos(thetas[i]) * sigmaz()
        H = -H             #Figure this out please
        _, eigs = np.linalg.eigh(H)
        eigs=eigs.T        # the eigenvectors need to be rows in eigfs
        eigfs[i, j, :, :] = eigs[:, :]
b_curv = berry_curvature(eigfs)
plot_berry_curvature(eigfs)

print('The Chern number is:')
print(b_curv.sum()/2/np.pi )
print('The chern nuber of two filled bands is 0. Nonzero chern number can be found in the gap. The only gap is between the two bands.')
print('To calculate the chern numbers of 1 filled band, the number of occupied band(max_occ) would be 1.')


max_occ = 1
occ_bnds = np.zeros((N,N,max_occ,2),dtype=complex)
for i in range(max_occ):
    occ_bnds[:,:,i,:] = eigfs[:,:,i,:]

print(np.shape(occ_bnds))  

b_curv = berry_curvature(occ_bnds)
plot_berry_curvature(occ_bnds)
print('The Chern number is:')
print(b_curv.sum()/2/np.pi )


print('All columns of b_curv should be proportional to sin(\theta)')
sinn = b_curv[:,1]
fig, ax = subplots()
ax.plot(thetas[0:N-1], b_curv[:,5]);
show()

print('All rows of b_curv should be constant')
fig, ax = subplots()
ax.plot(thetas[0:N-1], b_curv[20,:]);
show()

