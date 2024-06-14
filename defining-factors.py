import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

def sig(x):
    return 1/(1 + np.exp(-x))

def dedt(E):
    de = (-E + sig(k*E + k*P)) / tau_E
    return de

def ddedt(E):
    s =  sig(k*E + k*P)
    dde = (-1 + k * s * (1 - s)) / tau_E
    return dde

def h(x):
    x = np.asarray(x)
    return x + dedt(x) 

def j_func(x):
    x = np.asarray(x)
    return 1 + ddedt(x)

def gauss_factor(x0):
    sigma_inv = 1/sigma_meas
    j_x0 = j_func(x0)

    eta = j_x0.T * sigma_inv * (z_x0 - h(x0) + j_x0 * x0)
    lam = j_x0.T * sigma_inv * j_x0

    mu = eta / lam
    sig = 1/lam

    return mu, sig

def gauss1d(mean, std):
    return norm.pdf(X, mean, std)

if __name__ == "__main__":
    k = 5.
    tau_E = 1.
    P = .2
    X = np.linspace(-1.5, 1.5, 500)
    sigma_meas = 0.3

    x0 = 0.2
    z_x0 = h(x0)

    mu, std = gauss_factor(x0)
    print(z_x0, mu, std)
    
    plt.plot(X, h(X))
    plt.show()
