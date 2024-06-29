import torch
from torch._prims_common import Tensor
from typing import Tuple, Union

class Gaussian:
    '''
    Defines a Gaussian in moments form with mu and cov
    '''
    def __init__(self, mu  : Tensor, cov : Tensor) -> None:
        self._eta, self._lmbda = self.moments_to_canonical(mu, cov)

    @classmethod
    def from_canonical(cls, eta: Tensor, lmbda: Tensor) -> 'Gaussian':
        '''
        Alternative constructor to create a Gaussian from canonical parameters.
        '''
        g = cls.__new__(cls)
        g._eta = eta
        g._lmbda = lmbda
        return g

    @classmethod
    def zeros_like(cls, other) -> 'Gaussian':
        g = cls.__new__(cls)
        g._eta = torch.zeros_like(other.eta)
        g._lmbda = torch.zeros_like(other.lmbda)
        return g

    @property
    def mean(self) -> Tensor:
        return torch.linalg.inv(self._lmbda) @ self._eta

    @property
    def cov(self) -> Tensor:
        return torch.linalg.inv(self._lmbda)

    @property
    def eta(self) -> Tensor:
        return self._eta

    @property
    def lmbda(self) -> Tensor:
        return self._lmbda

    @staticmethod
    def canonical_to_moments(eta: Tensor, lmbda: Tensor) -> Tuple[Tensor, Tensor]:
        cov = torch.linalg.inv(lmbda)
        mu = cov @ eta
        return cov, mu

    @staticmethod
    def moments_to_canonical(mu: Tensor, cov: Tensor) -> Tuple[Tensor, Tensor]:
        lmbda = torch.linalg.inv(cov)
        eta = lmbda @ mu
        return eta, lmbda

    def clone(self) -> 'Gaussian':
        return Gaussian(self._eta.clone(), self._lmbda.clone())

    def __mul__(self, other: Union['Gaussian', float]) -> 'Gaussian':
        if isinstance(other, Gaussian):
            return Gaussian.from_canonical(self._eta + other._eta, self._lmbda + other._lmbda)
        elif isinstance(other, float):
            return Gaussian.from_canonical(self._eta * other, self._lmbda * other)
        else:
            raise TypeError("Multiplication is only supported between Gaussian objects or with a float")

    def __imul__(self, other: Union['Gaussian', float]) -> 'Gaussian':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'Gaussian':
        return Gaussian(self._eta - other._eta, self._lmbda - other._lmbda)

    def __idiv__(self, other) -> 'Gaussian':
        return self.__truediv__(other)

    def __str__(self):
        return f'[eta={self._eta}, lambda={self._lmbda}]'
