from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch._prims_common import Tensor
from utils import Gaussian, canonical_to_moments, sig, simulate_signal
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

        self.inbox = defaultdict(lambda: Gaussian(0., 0.))
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

            # Eqn. 2.21 from Ortiz (2023), which is var -> factor message
            in_eta, in_lmbda = self.inbox[fid].eta, self.inbox[fid].lmbda
            out_eta, out_lmbda = self.eta - in_eta, self.lmbda - in_lmbda

            factor_nodes[fid].inbox[self.var_id] = Gaussian(out_eta, out_lmbda)

        self.inbox.clear()

    def __str__(self):
        return f'Variable node {self.var_id} connected to {[self.prior_id, self.left_id, self.right_id]}'


class Parameter:
    def __init__(self, param_id, mu, sigma, connected_factors):
        self.param_id = param_id
        self.mu = mu
        self.sigma = sigma
        self.connected_factors = connected_factors

        self.lmbda = torch.linalg.inv(sigma)
        self.eta = self.lmbda @ mu

        self.inbox = {}

    def get_eta(self): return self.eta

    def get_lmbda(self): return self.lmbda

    def get_mu(self): return self.mu

    def get_sigma(self): return self.sigma

    def send_initial_message(self):
        for fid in self.connected_factors:
            factor_nodes[fid].inbox[self.param_id] = Gaussian(self.eta, self.lmbda)

    def belief_update(self):
        eta, lmbda = torch.zeros_like(self.eta), torch.zeros_like(self.lmbda)

        # Consume messages from the inbox and update belief
        for _, message in self.inbox.items():
            eta += message.eta
            lmbda += message.lmbda

        if lmbda == 0. : print('We Hebben Een Serieus Probleem in le parameter')

        self.eta, self.lmbda = eta, lmbda

        self.sigma = torch.linalg.inv(self.lmbda)
        self.mu = self.sigma @ self.eta

    def compute_messages(self):
        self.belief_update()

        for fid in self.connected_factors:
            if fid == -1: continue

            # Eqn. 2.21 from Ortiz (2023), which is var -> factor message
            in_eta, in_lmbda = self.inbox[fid].eta, self.inbox[fid].lmbda
            out_eta, out_lmbda = self.eta - in_eta, self.lmbda - in_lmbda

            factor_nodes[fid].inbox[self.param_id] = Gaussian(out_eta, out_lmbda)

        self.inbox.clear()

    def __str__(self):
        return f'Parameter {self.param_id} connected to {self.connected_factors}'


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
        # r = self.z - var_nodes[self.var_id].get_mu()
        # M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        # # Eqn. 3.20 from Ortiz (2023)
        # if M > self.N_sigma:
        #     kR = (2*self.N_sigma / M) - (self.N_sigma**2 / M**2)
        #     return kR

        return 1

    def compute_messages(self):
        kR = self.huber()

        # Add to the inbox of the variable attached to this factor
        var_nodes[self.var_id].inbox[self.f_id] = Gaussian(self.get_eta() * kR, self.get_lmbda() * kR)

    def __str__(self):
        return f'Observation factor with z = {self.z} connected to {self.var_id}'


class DynamicsFactor:
    def __init__(self, Et_id, Etp_id, lmbda_in, f_id, k_id) -> None:
        self.Et_id = Et_id
        self.Etp_id = Etp_id
        self.k_id = k_id
        self.lmbda_in = lmbda_in
        self.f_id = f_id

        self.out_eta, self.out_lmbda = torch.tensor([[0.]]), torch.tensor([[0.]])
        self.N_sigma = torch.sqrt(lmbda_in[0,0])
        self.z = 0

        self._prev_messages = defaultdict(lambda: Gaussian(0., 0.))
        self.inbox = defaultdict(lambda: Gaussian(0., 0.))

        self._vars = [Et_id, Etp_id, k_id]
        # self._vars = [Et_id, Etp_id]


    def linearise(self):
        # Bunch of computations to linearise this factor (Ortiz (2023) eqns. 2.46 and 2.47)
        Et_mu  = var_nodes[self.Et_id].get_mu()
        Etp_mu = var_nodes[self.Etp_id].get_mu()
        k_mu   = var_nodes[self.k_id].get_mu()

        Et_mu  = Et_mu.clone().detach().requires_grad_(True)
        Etp_mu = Etp_mu.clone().detach().requires_grad_(True)
        k_mu   = k_mu.clone().detach().requires_grad_(True)

        self.h = torch.abs(Etp_mu - (Et_mu + dt * dedt(Et_mu, k = k_mu)))
        # self.h = Etp_mu - Et_mu
        self.h.backward()

        J = torch.tensor([[Et_mu.grad, Etp_mu.grad, k_mu.grad]])
        # J = torch.tensor([[Et_mu.grad, Etp_mu.grad]])
        x0 = torch.tensor([Et_mu.item(), Etp_mu.item(), k_mu.item()])
        # x0 = torch.tensor([Et_mu.item(), Etp_mu.item()])

        self.eta = J.T @ self.lmbda_in * (J @ x0 - self.h)
        self.lmbda = (J.T @ self.lmbda_in) @ J

    def get_eta(self):
        return self.out_eta

    def get_lmbda(self):
        return self.out_lmbda

    def huber(self):
        # self.linearise()
        # r = self.z - self.h
        # M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        # # Eqn. 3.20 from Ortiz (2023)
        # if M > self.N_sigma:
        #     kR = (2*self.N_sigma / M) - (self.N_sigma**2 / M**2)
        #     return kR

        return 1

    def _compute_message_to_i(self, i, beta = .8):
        self.linearise()

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

        # All of this is eqn 8 from 'Learning in Deep Factor Graphs with Gaussian Belief Propagation'
        eta_i, eta_not_i = eta_here[i], eta_here[not_i_mask]
        lambda_ii = lambda_here[i,i]
        lambda_i_not_i = lambda_here[i, not_i_mask]
        lambda_not_i_i = lambda_here[not_i_mask, i]
        lambda_not_i_not_i = lambda_here[not_i_mask][:, not_i_mask]

        sigma_not_i_not_i = torch.linalg.inv(lambda_not_i_not_i)

        intermediate = lambda_i_not_i @ sigma_not_i_not_i
        lmbda = lambda_ii - intermediate @ lambda_not_i_i
        eta = eta_i - intermediate @ eta_not_i

        # Apply message damping
        damped_eta = beta * eta + (1 - beta) * self._prev_messages[i].eta
        damped_lmbda = beta * lmbda + (1 - beta) * self._prev_messages[i].lmbda

        # Store previous message
        self._prev_messages[i] = Gaussian(eta, lmbda)

        return Gaussian(damped_eta, damped_lmbda)

    def compute_messages(self):
        for i, item in enumerate(self._vars):
            var_nodes[item].inbox[self.f_id] = self._compute_message_to_i(i)

        self.inbox = {}

    def __str__(self):
        return f'Dynamics factor {self.f_id} connecting {self._vars}'


def print_fg(vars, factors):
    for i in vars:
        print(vars[i])

    for i in factors:
        print(factors[i])

def update_observational_factor(key):
    if not isinstance(key, tuple):
        factor_nodes[key].compute_messages()

def update_variable_belief(key):
    if key != k_id: var_nodes[key].compute_messages()

def update_dynamics_factor(key):
    if isinstance(key, tuple):
        factor_nodes[key].compute_messages()

if __name__ == "__main__":
    sigma_obs = 1e-2
    sigma_dynamics = 1e-3

    GT = 0.8
    signal = simulate_signal(15, 0.01, GT, 0.2, 1.)[250:]
    signal += torch.normal(0, 0.1, size=signal.shape)

    # signal = [1., 2., 3., 4.]
    t = torch.arange(0, len(signal), 1)
    dt = 0.01
    iters = 50

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    ax = axs

    k_id = len(t)
    k_param = Parameter(k_id, torch.tensor([[.5]]), torch.tensor([[2.]]), [])

    var_nodes = {}
    factor_nodes = {}

    # == CONSTRUCT FG === #
    for i in range(len(t)):
        var_nodes[i] = Variable(i, torch.tensor([[0.]]), torch.tensor([[0.1]]), -1 if i == 0 else (i-1, i), -1 if i+1 == len(t) else (i,i+1), i)
        factor_nodes[i] = ObservationFactor(i, i, signal[i], torch.tensor([[sigma_obs ** -2]]))

    for i in range(len(t)):
        if i + 1 < len(t):
            dyn_id = (i, i+1)
            factor_nodes[dyn_id] = DynamicsFactor(i, i+1, torch.tensor([[sigma_dynamics ** -2]]), dyn_id, k_id)
            k_param.connected_factors.append(dyn_id)

    var_nodes[k_id] = k_param

    # === RUN GBP (Simultaneous Schedule)=== #
    # for i in range(iters):
    #     # print(f'---- Iteration {i} ----')
    #     if i == 0: var_nodes[k_id].send_initial_message()

    #     for key in factor_nodes:
    #         update_observational_factor(key)

    #     for key in var_nodes:
    #         update_variable_belief(key)

    #     for key in factor_nodes:
    #         update_dynamics_factor(key)

    #     var_nodes[k_id].compute_messages()


    # == RUN GBP (Sweep schedule) === #
    for i in range(iters):
        if i == 0: var_nodes[k_id].send_initial_message()

        for key in factor_nodes:
            update_observational_factor(key)

        # -- RIGHT PASS -- #
        for i in range(len(t)):
            # -- Update variable belief and send message right -- #
            curr = var_nodes[i]

            curr.belief_update()

            if curr.right_id == -1: continue

            in_eta, in_lmbda = curr.inbox[curr.right_id].eta, curr.inbox[curr.right_id].lmbda
            out_eta, out_lmbda = curr.get_eta() - in_eta, curr.get_lmbda() - in_lmbda

            factor_nodes[curr.right_id].inbox[i] = Gaussian(out_eta, out_lmbda)

            # -- Update dynamics factor and send message right -- #
            fac = factor_nodes[curr.right_id]

            var_nodes[fac.Etp_id].inbox[fac.f_id] = fac._compute_message_to_i(1)
            var_nodes[fac.k_id].inbox[fac.f_id] = fac._compute_message_to_i(2)

        var_nodes[k_id].compute_messages()

        # -- LEFT PASS -- #
        for i in range(len(t)-1, -1, -1):
            # -- Update variable belief and send message left -- #
            curr = var_nodes[i]

            curr.belief_update()

            if curr.left_id == -1: continue

            in_eta, in_lmbda = curr.inbox[curr.left_id].eta, curr.inbox[curr.left_id].lmbda
            out_eta, out_lmbda = curr.get_eta() - in_eta, curr.get_lmbda() - in_lmbda

            factor_nodes[curr.left_id].inbox[i] = Gaussian(out_eta, out_lmbda)

            # -- Update dynamics factor and send message left -- #
            fac = factor_nodes[curr.left_id]

            var_nodes[fac.Et_id].inbox[fac.f_id] = fac._compute_message_to_i(0)
            var_nodes[fac.k_id].inbox[fac.f_id] = fac._compute_message_to_i(2)

        var_nodes[k_id].compute_messages()


    # == Extract and plot === #
    recons_signal = torch.tensor([var_nodes[key].get_mu() for key in var_nodes if key != k_id])
    print(recons_signal)
    print(var_nodes[k_id].get_mu().item(), var_nodes[k_id].get_sigma().item())
    ax.plot(recons_signal, label = rf'k = {var_nodes[k_id].get_mu().item()}')

    ax.set_title(rf'$\sigma_o = {sigma_obs}, \sigma_d = {sigma_dynamics}$')

    ax.plot(signal, label = f'Noisy Signal, k = {GT}')
    ax.legend()

    # plt.tight_layout()
    # plt.savefig('temp4.pdf', dpi=600, bbox_inches='tight')
    plt.show()
