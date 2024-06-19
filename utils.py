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
