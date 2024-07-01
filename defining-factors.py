from collections import defaultdict
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch._prims_common import Tensor
from utils import Gaussian, canonical_to_moments, sig, simulate_signal, softplus
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
        return f'Parameter {self.param_id} with mu = {self.mu} and sigma = {self.sigma}'


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

    def __str__(self):
        return f'Observation factor with z = {self.z} connected to {self.var_id}'


class DynamicsFactor:
    def __init__(self, Et_id, Etp_id, lmbda_in, f_id, parameters) -> None:
        self.Et_id = Et_id
        self.Etp_id = Etp_id
        self.parameters = parameters
        self.lmbda_in = lmbda_in
        self.f_id = f_id

        self.out_eta, self.out_lmbda = torch.tensor([[0.]]), torch.tensor([[0.]])
        self.N_sigma = torch.sqrt(lmbda_in[0,0])
        self.z = 0

        self._prev_messages = defaultdict(lambda: Gaussian(0., 0.))
        self.inbox = defaultdict(lambda: Gaussian(0., 0.))

        self._vars = [Et_id, Etp_id] + parameters


    def linearise(self):
        # Bunch of computations to linearise this factor (Ortiz (2023) eqns. 2.46 and 2.47)
        connected_variables = [var_nodes[i].get_mu().clone().detach().requires_grad_(True) for i in self._vars]
        Et_mu, Etp_mu = connected_variables[0], connected_variables[1]

        self.h = torch.abs(Etp_mu - (Et_mu + dt * dedt(Et_mu, *connected_variables[2:])))
        # self.h = Etp_mu - Et_mu
        self.h.backward()

        J = torch.tensor([[v.grad for v in connected_variables]])
        x0 = torch.tensor([v.item() for v in connected_variables])

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

    def _compute_message_to_i(self, i, beta = 0.8):
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


    def compute_messages_right(self):
        self.linearise()

        right_factors = self._vars[1:]
        for id in right_factors:
            var_nodes[id].inbox[self.f_id] = self._compute_message_to_i(self._vars.index(id))


    def compute_messages_left(self):
        self.linearise()

        left_factors = [self._vars[0]] + self._vars[2:]
        for id in left_factors:
            var_nodes[id].inbox[self.f_id] = self._compute_message_to_i(self._vars.index(id))

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
    if key not in param_ids: var_nodes[key].compute_messages()

def update_dynamics_factor(key):
    if isinstance(key, tuple):
        factor_nodes[key].compute_messages()

def update_params():
    for p in param_ids:
        var_nodes[p].compute_messages()


if __name__ == "__main__":
    # Constants for simulation
    sigma_obs = 1e-2
    sigma_dynamics = 1e-3
    GT_k = 2.5
    T, dt = 15, 0.01
    iters = 60

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # signal = simulate_signal(T, dt, GT_k)[250:]
    # noise  = torch.normal(0, 0, size=signal.shape)
    # signal += noise
    signal = [1., 2., 3., 4.]
    t = torch.arange(0, len(signal), 1)


    ax = axs

    k_id, p_id = range(len(t), len(t) + 2)
    params = {
        'k': Parameter(k_id, torch.tensor([[3.0]]), torch.tensor([[2.]]), []),
        # 'p': Parameter(p_id, torch.tensor([[.5]]), torch.tensor([[2.]]), [])
    }

    var_nodes = {}
    factor_nodes = {}
    param_ids = [p.param_id for _,p in params.items()]

    # == CONSTRUCT FG === #
    for i in range(len(t)):
        var_nodes[i] = Variable(i, torch.tensor([[0.]]), torch.tensor([[0.1]]), -1 if i == 0 else (i-1, i), -1 if i+1 == len(t) else (i,i+1), i)
        factor_nodes[i] = ObservationFactor(i, i, signal[i], torch.tensor([[sigma_obs ** -2]]))

    for i in range(len(t)):
        if i + 1 < len(t):
            dyn_id = (i, i+1)
            factor_nodes[dyn_id] = DynamicsFactor(i, i+1, torch.tensor([[sigma_dynamics ** -2]]), dyn_id, param_ids)

            for _,p in params.items():
                p.connected_factors.append(dyn_id)

    for _,p in params.items():
        var_nodes[p.param_id] = p

    # == RUN GBP (Sweep schedule) === #
    for i in range(iters):
        print(f'-- Iteration {i}, currently at k = {var_nodes[k_id].get_mu().item():4f} sigma={var_nodes[k_id].get_sigma().item():4f} --')
        if i == 0:
            for p in param_ids:
                var_nodes[p].send_initial_message()

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

            fac.compute_messages_right()

        update_params()

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

            fac.compute_messages_left()

        update_params()

    ax.plot(signal, label = rf'Noisy Signal')

    # == Extract and plot === #
    recons_signal = torch.tensor([var_nodes[key].get_mu() for key in var_nodes if key not in param_ids])
    print(recons_signal)
    ax.plot(recons_signal, label = rf'Reconstructed Mus')

    rec = simulate_signal(T, dt, *[p.get_mu().item() for _,p in params.items()])[250:]
    # rec += noise
    ax.plot(rec, label=f'Reconstructed Simuluation')

    for _,p in params.items():
        print(p)

    ax.set_title(rf'$\sigma_o = {sigma_obs}, \sigma_d = {sigma_dynamics}$')

    ax.legend()

    plt.tight_layout()
    # plt.savefig('temp4.pdf', dpi=600, bbox_inches='tight')
    plt.show()
