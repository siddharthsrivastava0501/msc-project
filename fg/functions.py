from torch import Tensor
import torch

def sig(x) -> Tensor:
    return 1/(1+torch.exp(-x))

def tanh(x) -> Tensor:
    return 2*sig(2*x) - 1

def dEdt(E, I, k1 = 10., k2 = 12., P = 0.2, tau_E = 1., act = sig) -> Tensor:
    de = (-E + act(k1*E - k2*I + P)) / tau_E
    return de

def dIdt(E, I, k3 = 9., k4 = 3., Q = 0.2, tau_I = 2., act = sig) -> Tensor:
    di = (-I + act(k3*E - k4*I + Q)) / tau_I
    return di
