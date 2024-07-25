# Code from
# https://gist.github.com/Geometrein/bc15807e5e9dd2e94fe60e2c7a4cd030

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fg.functions import pairwise_difference_matrix

def hopf(X, Y, a, omega, v, beta = 0.):
    """
    Hopf equation from Equation 4 and 5 of Deco & Kringelbach, 2020
    "Turbulent-like Dynamics in the Human Brain"
    """
    noise = v * np.random.normal(size=X.shape)
    square_term = X**2 + Y**2
    dX = a*X - omega*Y + square_term*(beta*Y - X) + noise
    dY = a*Y + omega*X + square_term*(beta*X + Y) + noise
    return dX, dY

# Constants
dt = 0.01
x_init = 0.3
y_init = 0.2
T = 10
n_regions = 100

times = np.arange(0, T + dt, dt)
X = np.zeros((n_regions, len(times)))
Y = np.zeros((n_regions, len(times)))
X[0] = x_init
Y[0] = y_init

a = np.full((n_regions, ), -0.02)
omega = np.full((n_regions, ), 0.1)
v = np.full((n_regions, ), 0.01)
beta = 0.1
G = 0.8

# Generate a random symmetric connectivity matrix
C = pd.read_csv('./fmri/DTI_fiber_consensus_HCP.csv', header=None).to_numpy()[:n_regions, :n_regions]
C /= C.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(C, 0)

print(X[:, 0].size)

for t in range(len(times)-1):
    X_coupling = G * np.sum(C @ pairwise_difference_matrix(X[:, t]), axis=0)
    Y_coupling = G * np.sum(C @ pairwise_difference_matrix(Y[:, t]), axis=0)

    dX, dY = hopf(X[:, t], Y[:, t], a, omega, v, beta)
    X[:, t+1] = X[:, t] + dt * (dX + X_coupling)
    Y[:, t+1] = Y[:, t] + dt * (dY + Y_coupling)

# Fig. 1: X as a function of time
plt.figure()
# plt.plot(X[0])
plt.plot(X[1])
# plt.plot(X[3])
# plt.plot(X[4])
plt.xlabel("Time")
plt.ylabel("X")
plt.show()
