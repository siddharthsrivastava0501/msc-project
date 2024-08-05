import torch
from torch import Tensor
from .functions import dEdt, dIdt, sig
import numpy as np

def _initial_C(nr):
    C = torch.empty((nr, nr)).normal_(0.2, 0.1)
    C.fill_diagonal_(0.)

    return C

def simulate_wc(config : dict) -> tuple[Tensor, Tensor]:
    '''
    Simulates the excitatory and inhibitory dynamics of Wilson-Cowan using the
    equations found here:
    https://en.wikipedia.org/wiki/Wilson–Cowan_model#Simplification_of_the_model_assuming_time_coarse_graining
    '''
    T = config.get('T', 6.)
    dt = config.get('dt', 0.01)
    nr = config.get('nr', 5)

    torch.empty((nr,)).normal_(5.)

    a = config.get('a', torch.empty((nr,)).normal_(3., 1.))
    b = config.get('b', torch.empty((nr,)).normal_(5., 1.))
    c = config.get('c', torch.empty((nr,)).normal_(4., 1.))
    d = config.get('d', torch.empty((nr,)).normal_(3., 1.))
    P = config.get('P', torch.empty((nr,)).normal_(1., 0.2))
    Q = config.get('Q', torch.empty((nr,)).normal_(1., 0.2))
    tauE = config.get('tauE', torch.full((nr,), 1.))
    tauI = config.get('tauI', torch.full((nr,), 2.))
    C = config.get('C', _initial_C(nr))
    
    simulation_info = (
        f"Running simulation with: "
        f"T = {T}, dt = {dt}, nr = {nr}, a = {a}, b = {b}, c = {c}, d = {d}, "
        f"P = {P}, Q = {Q}, tauE = {tauE}, tauI = {tauI}, nr = {nr}"
    )
    print(simulation_info)

    time = torch.arange(0, T + dt, dt)
    E = np.zeros((len(time), nr))
    I = np.zeros((len(time), nr))

    E[0] = 0.3
    I[0] = 0.4

    for t in range(len(time) - 1):
        E_input = np.dot(C, E[t])
        I_input = np.dot(C, I[t])

        for r in range(nr):
            E[t+1, r] = E[t, r] + dt * dEdt(E[t, r], I[t, r], E_input[r], a[r], b[r], P[r], tauE[r])
            I[t+1, r] = I[t, r] + dt * dIdt(E[t, r], I[t, r], I_input[r], c[r], d[r], Q[r], tauI[r])

    return E, I

def simulate_stuart_landau(config : dict):
    np.random.seed(42)

    T = config.get('T', 10.)
    dt = config.get('dt', 0.01)
    a = config.get('a', -0.02)
    omega = config.get('omega', 0.1)
    beta = config.get('beta', 0.02)

    times = torch.arange(0, T + dt, dt)
    X = torch.zeros(len(times))
    Y = torch.zeros(len(times))

    for t in range(len(times) - 1):
        noise = np.random.normal(0, beta)
        X[t+1] = X[t] + dt * ((a - X[t]**2 - Y[t]**2) * X[t] - omega * Y[t] + noise)
        Y[t+1] = Y[t] + dt * ((a - X[t]**2 - Y[t]**2) * Y[t] + omega * X[t] + noise)

    return X, Y