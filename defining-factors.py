import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch._prims_common import Tensor
from utils import Gaussian, canonical_to_moments, sig
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

var_nodes, factor_nodes = {}, {}

def dedt(E, k, P = 0.2, tau_E = 1.):
    de = (-E + sig(k*E + k*P)) / tau_E
    return de

class Variable:
    def __init__(self, var_id, mu, sigma, left_id, right_id, prior_id) -> None:
        self.var_id = var_id
        self.mu = mu
        self.sigma = sigma
        self.left_id  = left_id
        self.right_id = right_id
        self.prior_id = prior_id

        self.eta = torch.zeros_like(mu)
        self.lmbda = torch.zeros_like(sigma)

        self.inbox = {}
        self.factors = [prior_id, left_id, right_id]

    def get_eta(self): return self.eta

    def get_lmbda(self): return self.lmbda

    def get_mu(self): return self.mu

    def get_sigma(self): return self.sigma

    def belief_update(self):
        '''
        Belief update for a variable node is simply the product of all received messages, Ortiz (2023) eq. 2.13
        '''
        eta, lmbda = torch.zeros_like(self.eta), torch.zeros_like(self.lmbda)

        # Consume messages from the inbox and update belief
        for _, message in self.inbox.items():
            eta += message.eta
            lmbda += message.lmbda

        if lmbda == 0. : print('We Hebben Een Serieus Probleem ')

        self.eta, self.lmbda = eta, lmbda

        self.sigma = torch.linalg.inv(self.lmbda)
        self.mu = self.sigma @ self.eta

    def compute_messages(self):
        self.belief_update()

        for fid in self.factors:
            if fid == -1: continue

            try:
                # Eqn. 2.21 from Ortiz (2023), which is var -> factor message
                in_eta, in_lmbda = self.inbox[fid].eta, self.inbox[fid].lmbda
                out_eta, out_lmbda = self.eta - in_eta, self.lmbda - in_lmbda

                factor_nodes[fid].inbox[self.var_id] = Gaussian(out_eta, out_lmbda)
            except KeyError:
                # print(f'Landed here, sending {self.eta} and {self.lmbda} to {fid}')
                factor_nodes[fid].inbox[self.var_id] = Gaussian(self.eta, self.lmbda)

        self.inbox = {}

class Parameter:
    def __init__(self, param_id, mu, sigma, connected_factors):
        self.param_id = param_id
        self.mu = mu
        self.sigma = sigma
        self.connected_factors = connected_factors

        self.lmbda = torch.linalg.inv(self.sigma)
        self.eta = self.lmbda @ self.mu

        self.inbox = {}

    def get_eta(self): return self.eta

    def get_lmbda(self): return self.lmbda

    def get_mu(self): return self.mu

    def get_sigma(self): return self.sigma

    def send_initial_messages(self):
        for fid in self.connected_factors:
            if fid == -1: continue

            factor_nodes[fid].inbox[self.param_id] = Gaussian(self.eta, self.lmbda)

    def belief_update(self):
        '''
        Belief update for a variable node is simply the product of all received messages, Ortiz (2023) eq. 2.13
        '''
        eta, lmbda = torch.zeros_like(self.eta), torch.zeros_like(self.lmbda)

        # Consume messages from the inbox and update belief
        for _, message in self.inbox.items():
            eta += message.eta
            lmbda += message.lmbda

        if lmbda == 0. : print('We Hebben Een Serieus Probleem ')

        self.eta, self.lmbda = eta, lmbda

        self.sigma = torch.linalg.inv(self.lmbda)
        self.mu = self.sigma @ self.eta

    def compute_messages(self):
        self.belief_update()

        for fid in self.connected_factors:
            if fid == -1: continue

            try:
                # Eqn. 2.21 from Ortiz (2023), which is var -> factor message
                in_eta, in_lmbda = self.inbox[fid].eta, self.inbox[fid].lmbda
                out_eta, out_lmbda = self.eta - in_eta, self.lmbda - in_lmbda

                factor_nodes[fid].inbox[self.param_id] = Gaussian(out_eta, out_lmbda)
            except KeyError:
                # print(f'Landed here, sending {self.eta} and {self.lmbda} to {fid}')
                factor_nodes[fid].inbox[self.param_id] = Gaussian(self.eta, self.lmbda)

        self.inbox = {}


class ObservationFactor:
    def __init__(self, f_id, var_id, z, lmbda_in) -> None:
        self.f_id = f_id
        self.var_id = var_id
        self.z = z
        self.lmbda_in = lmbda_in

        J = torch.tensor([[1.]])
        self.eta = (J.T @ lmbda_in) * z
        self.lmbda = (J.T @ lmbda_in) @ J
        self.N_sigma = torch.sqrt(self.lmbda_in[0,0])

        self.out_eta, self.out_lmbda = self.eta, self.lmbda
        self.inbox = {}

    def get_eta(self):
        return self.out_eta

    def get_lmbda(self):
        return self.out_lmbda

    def huber(self):
        r = self.z - var_nodes[self.var_id].get_mu()
        M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        # Eqn. 3.20 from Ortiz (2023)
        if M > self.N_sigma:
            kR = (2*self.N_sigma / M) - (self.N_sigma**2 / M**2)
            return kR

        return 1

    def compute_messages(self):
        kR = self.huber()

        # Add to the inbox of the variable attached to this factor
        var_nodes[self.var_id].inbox[self.f_id] = Gaussian(self.get_eta() * kR, self.get_lmbda() * kR)


class DynamicsFactor:
    def __init__(self, Et_id, Etp_id, lmbda_in, f_id, k_id) -> None:
        self.Et_id = Et_id
        self.Etp_id = Etp_id
        self.lmbda_in = lmbda_in
        self.f_id = f_id

        self.out_eta, self.out_lmbda = torch.tensor([[0.]]), torch.tensor([[0.]])
        self.inbox = {}
        self.N_sigma = torch.sqrt(lmbda_in[0,0])
        self.z = 0
        self.k_id = k_id

        self._vars = [Et_id, Etp_id, k_id]

    def linearise(self):
        # Bunch of computations to linearise this factor (Ortiz (2023) eqns. 2.46 and 2.47)
        Et_mu, _  = canonical_to_moments(self.inbox[self.Et_id].eta, self.inbox[self.Et_id].lmbda)
        Etp_mu, _ = canonical_to_moments(self.inbox[self.Etp_id].eta, self.inbox[self.Etp_id].lmbda)
        # k_mu, _   = canonical_to_moments(self.inbox[self.k_id].eta, self.inbox[self.k_id].lmbda)

        Et_mu  = Et_mu.clone().detach().requires_grad_(True)
        Etp_mu = Etp_mu.clone().detach().requires_grad_(True)
        # k_mu   = k_mu.clone().detach().requires_grad_(True)
        k_mu = torch.tensor([[.5]], requires_grad=True)

        self.h = torch.abs(Etp_mu - (Et_mu + dt * dedt(Et_mu, k = k_mu)))
        # self.h = Etp_mu - Et_mu
        self.h.backward()

        J = torch.tensor([[Et_mu.grad, Etp_mu.grad, k_mu.grad]])
        x0 = torch.tensor([Et_mu, Etp_mu, k_mu])

        self.eta = J.T @ self.lmbda_in * (J @ x0 - self.h)
        self.lmbda = (J.T @ self.lmbda_in) @ J

    def get_eta(self):
        return self.out_eta

    def get_lmbda(self):
        return self.out_lmbda

    def huber(self):
        self.linearise()

        r = self.z - self.h
        M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        # Eqn. 3.20 from Ortiz (2023)
        if M > self.N_sigma:
            kR = (2*self.N_sigma / M) - (self.N_sigma**2 / M**2)
            return kR

        return 1

    def _compute_message_to_i(self, i):
        self.linearise()

        # All of this is eqn 8 from 'Learning in Deep Factor Graphs with Gaussian Belief Propagation'
        eta_here, lambda_here = self.eta.clone(), self.lmbda.clone()

        not_i_mask = torch.ones(eta_here.shape[0], dtype=torch.bool)
        not_i_mask[i] = False

        in_eta = torch.sum(torch.tensor([self.inbox[k].eta for j,k in enumerate(self._vars) if j != i]))
        in_lmbda = torch.sum(torch.tensor([self.inbox[k].lmbda for j,k in enumerate(self._vars) if j != i]))
        kR = self.huber()

        eta_here[not_i_mask] = self.eta[not_i_mask] + in_eta
        lambda_here[not_i_mask, not_i_mask] = self.lmbda[not_i_mask, not_i_mask] + in_lmbda

        # Huberify
        eta_here *= kR
        lambda_here *= kR

        eta_i, eta_not_i = eta_here[i], eta_here[not_i_mask]
        lambda_ii = lambda_here[i,i]
        lambda_i_not_i = lambda_here[i, not_i_mask]
        lambda_not_i_i = lambda_here[not_i_mask, i]
        lambda_not_i_not_i = lambda_here[not_i_mask][:, not_i_mask]

        sigma_not_i_not_i = torch.linalg.inv(lambda_not_i_not_i)

        intermediate = lambda_i_not_i @ sigma_not_i_not_i
        lmbda = lambda_ii - intermediate @ lambda_not_i_i
        eta = eta_i - intermediate @ eta_not_i

        return Gaussian(eta, lmbda)

    def compute_messages(self):
        for i, item in enumerate(self._vars):
            var_nodes[item].inbox[self.f_id] = self._compute_message_to_i(i)

        self.inbox = {}


def update_observational_factor(key):
    if not isinstance(key, tuple):
        factor_nodes[key].compute_messages()

def update_variable_belief(key):
    if key not in param_ids:
        var_nodes[key].compute_messages()

def update_dynamics_factor(key):
    if isinstance(key, tuple):
        factor_nodes[key].compute_messages()

def update_param(key):
    if key in param_ids:
        var_nodes[key].compute_messages()

if __name__ == "__main__":
    sigma_obs = 1e-2
    sigma_dynamics = 1e-3

    signal = np.load('E_synthetic_kdot5_Pdot2.npy')[50:250].astype(np.float32)
    # signal = [1., 2., 3.]
    t = torch.arange(0, len(signal), 1)
    dt = 0.01
    iters = 50

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    ax = axs
    var_nodes = {}
    factor_nodes = {}

    k_id = len(t)
    param_ids = [k_id]

    # == CONSTRUCT FG === #
    k_var = Parameter(k_id, torch.tensor([[-.5]]), torch.tensor([[0.1]]), [])

    for i in range(len(t)):
        var_nodes[i] = Variable(i, torch.tensor([[0.]]), torch.tensor([[0.]]), -1 if i == 0 else (i-1, i), -1 if i+1 == len(t) else (i,i+1), i)
        factor_nodes[i] = ObservationFactor(i, i, signal[i], torch.tensor([[sigma_obs ** -2]]))

    for i in range(len(t)):
        if i + 1 < len(t):
            dyn_id = (i, i+1)
            factor_nodes[dyn_id] = DynamicsFactor(i, i+1, torch.tensor([[sigma_dynamics ** -2]]), dyn_id, k_id)
            k_var.connected_factors.append(dyn_id)

    var_nodes[k_id] = k_var

    # === RUN GBP === #
    for i in range(iters):
        # print(f'---- Iteration {i} ----')
        if i == 0: var_nodes[k_id].send_initial_messages()

        for key in factor_nodes:
            update_observational_factor(key)

        for key in var_nodes:
            update_variable_belief(key)

        for key in factor_nodes:
            update_dynamics_factor(key)

        for key in var_nodes:
            update_param(key)

    # == Extract and plot === #

    recons_signal = torch.tensor([var_nodes[key].get_mu() for key in var_nodes if key not in param_ids])
    found_k = var_nodes[k_id].get_mu()

    print(var_nodes[k_id].get_mu(), var_nodes[k_id].get_sigma())

    ax.plot(recons_signal, label = f'k = {found_k}')

    ax.set_title(rf'$\sigma_o = {sigma_obs}, \sigma_d = {sigma_dynamics}$')

    ax.plot(signal, label = 'Noisy Signal')
    ax.legend()

    # plt.tight_layout()
    # plt.savefig('temp4.pdf', dpi=600, bbox_inches='tight')
    plt.show()
