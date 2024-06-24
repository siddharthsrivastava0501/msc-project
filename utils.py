import torch
from torch._prims_common import Tensor


class Gaussian:
    '''
    Defines a Gaussian with an eta and a lambda
    '''
    def __init__(self, eta, lmbda) -> None:
        self.eta   : Tensor = eta
        self.lmbda : Tensor = lmbda

    def __str__(self):
        return f'Guassian object with eta = {self.eta} and lambda = {self.lmbda}'

def sig(x) -> Tensor:
    return 1/(1+torch.exp(-x))


def canonical_to_moments(eta : Tensor, lmbda : Tensor) -> tuple[Tensor, Tensor]:
    '''
    Converts the canonical parameters of a Gaussian to its moments parameters
    '''
    sig = torch.linalg.inv(lmbda)
    mu = sig @ eta

    return (mu, sig)


def select_not_i(tensor, i) -> Tensor:
    '''
    Selects not the ith row and ith column from a tensor
    '''
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[i] = False
    return tensor[mask][:, mask]


def simulate_signal(T, dt, k, P, tau_E):
    t = torch.arange(0, T + dt, dt)
    E = torch.zeros(len(t))

    def dedt(t, E, I, P):
        # de = (-E + (1 - r*E)*sig(w_ee*E - w_ei*I + P)) / tau_E
        de = (-E + sig(k*E + k*P)) / tau_E
        return de

    for i in range(len(t) - 1):
        E[i + 1] = E[i] + dt * dedt(0, E[i], 0, P)

    return E
