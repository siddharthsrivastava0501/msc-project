import torch
from torch import Tensor

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


def simulate_signal(config : dict) -> tuple[Tensor, Tensor]:
    '''
    Simulates the excitatory and inhibitory dynamics of Wilson-Cowan using the
    equations found here:
    https://en.wikipedia.org/wiki/Wilson–Cowan_model#Simplification_of_the_model_assuming_time_coarse_graining
    '''
    T = config.get('T', 6.)
    dt = config.get('dt', 0.01)
    k1 = config.get('k1', 10.)
    k2 = config.get('k2', 12.)
    k3 = config.get('k3', 9.)
    k4 = config.get('k4', 3.)
    P = config.get('P', 0.2)
    Q = config.get('Q', 0.5)
    tauE = config.get('tauE', 1.)
    tauI = config.get('tauI', 2.)
    act = config.get('act', sig)

    simulation_info = (
        f"Running simulation with: "
        f"T = {T}, dt = {dt}, k1 = {k1}, k2 = {k2}, k3 = {k3}, k4 = {k4}, "
        f"P = {P}, Q = {Q}, tauE = {tauE}, tauI = {tauI}, act = {act.__name__}"
    )
    print(simulation_info)

    t = torch.arange(0, T + dt, dt)
    E = torch.zeros(len(t))
    I = torch.zeros(len(t))

    for i in range(len(t) - 1):
        E[i + 1] = E[i] + dt * dEdt(E[i], I[i], k1, k2, P, tauE, act)
        I[i + 1] = I[i] + dt * dIdt(E[i], I[i], k3, k4, Q, tauI, act)

    return E, I
