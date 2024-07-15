# Code from
# https://gist.github.com/Geometrein/bc15807e5e9dd2e94fe60e2c7a4cd030

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def hopf(x, y, a, omega, v, beta = 0.):
    """
    Hopf equation from Equation 4 and 5 of Deco & Kringelbach, 2020
    "Turbulent-like Dynamics in the Human Brain"
    """
    square_term = (x**2 + y**2)
    noise = np.random.normal(0,1)
    dx = a*x + square_term*(beta * y - x) - omega*y + v*noise
    dy = a*y - square_term*(beta * x + y) - omega*x + v*noise

    return dx, dy

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

a = np.random.normal(-0.02, 0.01, (n_regions,))
omega = np.random.normal(0.1, 0.05, (n_regions,))
v = np.random.normal(0.01, 0.005, (n_regions,))
beta = 0.1
G = 0.8

# Generate a random symmetric connectivity matrix
C = pd.read_csv('./fmri/DTI_fiber_consensus_HCP.csv', header=None).to_numpy()[:n_regions, :n_regions]
C /= C.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(C, 0)

print(C.shape)

for t in range(len(times)-1):
    for n in range(n_regions):
        x_coupling = 0
        y_coupling = 0
        for p in range(n_regions):
            x_coupling += C[n,p]*(X[p, t] - X[n, t])
            y_coupling += C[n,p]*(Y[p, t] - Y[n, t])

        dx, dy = hopf(X[n, t], Y[n, t], a = a[n], omega = omega[n], v = v[n], beta = beta)

        X[n, t+1] = X[n, t] + dt * (dx + G * x_coupling)
        Y[n, t+1] = Y[n, t] + dt * (dy + G * y_coupling)

# Fig. 1: X as a function of time
plt.figure()
plt.plot(X[1])
plt.plot(X[2])
plt.plot(X[3])
plt.plot(X[4])
plt.xlabel("Time")
plt.ylabel("X")
plt.show()
