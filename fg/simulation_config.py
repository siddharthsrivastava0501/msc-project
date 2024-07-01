import torch
from torch._prims_common import Tensor

def sig(x) -> Tensor:
    return 1/(1+torch.exp(-x))

def dEdt(E, k = 1.2, P = 0.2, tau_E = 1.) -> Tensor:
    de = (-E + sig(k*E + k*P)) / tau_E
    return de

def simulate_signal(T, dt, k, P = 0.2, tau_E = 1.) -> Tensor:
    t = torch.arange(0, T + dt, dt)
    E = torch.zeros(len(t))

    for i in range(len(t) - 1):
        E[i + 1] = E[i] + dt * dEdt(E[i], k, P, tau_E)

    return E
