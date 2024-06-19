import numpy as np
import matplotlib.pyplot as plt
import torch

k = 5
P = .2
tau_E = 1

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
    
def _dedt(E):
    return (-E + sigmoid(k*E + k*P)) / tau_E

def meas_fn(Et, Etp):
    Etp, Et = torch.as_tensor(Etp), torch.as_tensor(Et)
    return torch.abs(Etp - (Et + _dedt(Et)))

Et, Etp = 1.5, 1.
Et_values = torch.linspace(-2, 2, 1000)
Etp_values = torch.linspace(-2, 2, 1000)

Et_tensor = torch.tensor(Et, requires_grad=True)
Etp_tensor = torch.tensor(Etp, requires_grad=True)

X0 = torch.tensor((Et_tensor, Etp_tensor))
h_X0 = meas_fn(Et_tensor, Etp_tensor)
h_X0.backward()
J = torch.tensor((Et_tensor.grad, Etp_tensor.grad))
# print(h_X0, J)

Et_grid, Etp_grid = torch.meshgrid(Et_values, Etp_values)

X_flat = torch.stack((Et_grid.flatten(), Etp_grid.flatten()), dim=1)

# print((X_flat - X0).shape)
h_approx_flat = h_X0 + J @ (X_flat - X0).T

meas_values = h_approx_flat.detach().view(Et_grid.shape).numpy()

plt.plot(Etp, Et, 'r.')
contour_levels = np.linspace(np.min(meas_values), np.max(meas_values), 100)
# # print(f'True Value: {meas_fn(0.2, 0.8)}, Approximated value: {h_approx(0.2, 0.8, 0.2, 0.8)}')
plt.contourf(Et_values, Etp_values, meas_values, levels=contour_levels )
plt.colorbar()
plt.show()