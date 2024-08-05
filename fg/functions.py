from torch import Tensor
import torch
import numpy as np

def sig(x, a = 1., thr = 0.):
    a, thr = torch.tensor(a), torch.tensor(thr)
    return 1 / (1 + torch.exp(-a * (x - thr)))

def pairwise_difference_matrix(x):
    '''
    Compute the difference matrix D of an input vector `x` = [x_1, ..., x_n]:
    D = [x_i - x_j] for all i,j.
    '''
    x = torch.as_tensor(x)

    x_col = x.view(-1, 1)
    x_row = x.view(1, -1)

    D = x_col - x_row

    return D.numpy()

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

def dEdt(Ei, Ii, E_ext, ai = 10., bi = 12., P = 0.2, tau_E = 1., act = Se, G = 0.8) -> Tensor:
    # de = (-E + (1-E)*act(k1*E - k2*I + P)) / tau_E
    de = (-Ei + act(ai*Ei - bi*Ii + P + G*E_ext)) / tau_E

    return de

def dIdt(Ei, Ii, I_ext, ci = 9., di = 3., Q = 0.2, tau_I = 2., act = Si, G = 0.8) -> Tensor:
    # di = (-I + (1-I)*act(k3*E - k4*I + Q)) / tau_I
    di = (-Ii + act(ci*Ei - di*Ii + Q - G*I_ext)) / tau_I

    return di


def hdEdt_dIdt(X, Y, a, omega, beta):
    '''
    We have to combine the dEdt and the dIdt for the Hopf model since they use
    the same shared noise?
    '''    
    noise = torch.normal(0, beta)
    dE = (a - X**2 - Y**2) * X - omega * Y + noise
    dI = (a - X**2 - Y**2) * Y + omega * X + noise
    
    return dE, dI

