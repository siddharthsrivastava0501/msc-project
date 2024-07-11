import torch
from torch import Tensor
from .functions import dEdt, dIdt, sig

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

    E[0] = 0.3
    I[0] = 0.4

    for i in range(len(t) - 1):
        E[i + 1] = E[i] + dt * dEdt(E[i], I[i], k1, k2, P, tauE, act)
        I[i + 1] = I[i] + dt * dIdt(E[i], I[i], k3, k4, Q, tauI, act)

    return E, I
