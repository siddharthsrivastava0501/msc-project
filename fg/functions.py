from torch import Tensor
import torch

def sig(x, a = 1., thr = 0.):
    a, thr = torch.tensor(a), torch.tensor(thr)
    return 1 / (1 + torch.exp(-a * (x - thr)))

def Se(x):
    aE = 1.3
    thrE = 4
    return sig(x, thrE, aE) - sig(0, thrE, aE)

def Si(x):
    aI = 2
    thrI = 3.7
    return sig(x, thrI, aI) - sig(0, thrI, aI)

def tanh(x) -> Tensor:
    return 2*sig(2*x) - 1

def dEdt(E, I, k1 = 10., k2 = 12., P = 0.2, tau_E = 1., act = Se) -> Tensor:
    de = (-E + (1-E)*act(k1*E - k2*I + P)) / tau_E
    return de

def dIdt(E, I, k3 = 9., k4 = 3., Q = 0.2, tau_I = 2., act = Si) -> Tensor:
    di = (-I + (1-I)*act(k3*E - k4*I + Q)) / tau_I
    return di
