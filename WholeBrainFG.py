import numpy as np
import matplotlib.pyplot as plt
import torch

def sig(x):
    return 1/(1 + torch.exp(-x))

def tanh(x):
    return 2*sig(2*x) - 1

def dedt(E):
    # de = (-E + (1 - r*E)*sig(w_ee*E - w_ei*I + P)) / tau_E
    de = (-E + sig(k*E + k*P)) / tau_E
    return de

var_nodes, factor_nodes = {}, {}

class Oscillator:
    def __init__(self, var_id, mu, sigma, left_dynamics_id, right_dynamics_id, obs_id):
        self.var_id = var_id
        self.mu = mu
        self.sigma = sigma
        self.eta = torch.zeros_like(mu, requires_grad=True)
        self.lmbda = torch.zeros_like(sigma, requires_grad=True)

        self.right_dynamics_id = right_dynamics_id
        self.left_dynamics_id = left_dynamics_id
        self.obs_id = obs_id

        self.factors = [self.obs_id, self.left_dynamics_id, self.right_dynamics_id]

        self.out_eta, self.out_lmbda = torch.zeros_like(mu), torch.zeros_like(sigma)

    def get_eta(self):
        return self.out_eta

    def get_mu(self):
        return self.mu

    def get_lmbda(self):
        return self.out_lmbda

    def get_sigma(self):
        return self.sigma

    def belief_update(self):
        eta_here, lmbda_here = torch.tensor([[0.]]), torch.tensor([[0.]])

        for fid in self.factors:
            if fid == -1: continue

            eta_here  += factor_nodes[fid].get_eta()
            lmbda_here += factor_nodes[fid].get_lmbda()

        if lmbda_here == 0.: print('We Hebben Een Serieus Probleem ')

        self.out_eta, self.out_lmbda = eta_here, lmbda_here

        self.sigma = torch.linalg.inv(self.out_lmbda)
        self.mu = self.sigma @ self.out_eta

    def compute_messages(self):
        self.belief_update()

        if self.right_dynamics_id == -1:
            self.out_eta = self.out_lmbda = 0
            return

        in_eta, in_lmbda = factor_nodes[self.right_dynamics_id].get_eta(), factor_nodes[self.right_dynamics_id].get_lmbda()

        self.out_eta -= in_eta
        self.out_lmbda -= in_lmbda


class ObservationFactor:
    def __init__(self, f_id, var_id, z, lmbda_in):
        self.f_id = f_id
        self.z = z
        self.lmbda_in = lmbda_in
        self.var_id = var_id

        # Jacobian
        J = torch.tensor([[1.]])

        self.eta = (J.T @ lmbda_in) * z
        self.lmbda = (J.T @ lmbda_in) @ J
        self.N_sigma = 0.8 / torch.sqrt(lmbda_in[0,0])

        self.out_eta, self.out_lmbda = self.eta, self.lmbda

    def get_eta(self):
        return self.out_eta

    def get_lmbda(self):
        return self.out_lmbda

    def huber_scale(self):
        r = self.z - var_nodes[self.var_id].get_mu()
        M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        if M > self.N_sigma:
            kR = (2 * self.N_sigma / M) - (self.N_sigma ** 2 / M ** 2)
            return kR

        return 1

    def compute_messages(self):
        kR = self.huber_scale()
        self.out_eta, self.out_lmbda = self.eta * kR, self.lmbda * kR

    def __str__(self):
        return f'Observation factor with connected to {self.var_id} and z = {self.z} '

class DynamicsFactor:
    ''' Et_id is the id of E_t, Etp_id is the id of E_{t+1}'''
    def __init__(self, f_id, lmbda_in, Et_id, Etp_id, beta = 1.):
        self.Et_id = Et_id
        self.Etp_id = Etp_id
        self.f_id = f_id
        self.lmbda_in = lmbda_in

        self.out_eta = torch.tensor([[0.0]])
        self.out_lmbda = torch.tensor([[0.0]])

        # Message damping
        self.prev_msg_right = None
        self.prev_msg_left = None
        self.beta = beta  # Damping constant

        self.N_sigma = torch.sqrt(lmbda_in[0,0])

        self.update_eta_lmbda()

    def update_eta_lmbda(self):
        Et = var_nodes[self.Et_id].get_mu()[0,0].clone().detach().requires_grad_(True)
        Etp = var_nodes[self.Etp_id].get_mu()[0,0].clone().detach().requires_grad_(True)
        self.h = Etp - (Et + dedt(Et))
        self.h.backward()

        J = torch.tensor([[Et.grad, Etp.grad]])

        x0 = torch.tensor([Et, Etp])
        self.eta = J.T @ self.lmbda_in * ((J @ x0) - self.h)
        self.lmbdap = (J.T @ self.lmbda_in) @ J


    def get_eta(self):
        return self.out_eta

    def get_lmbda(self):
        return self.out_lmbda

    def huber_scale(self):
        r = 0 - (self.h)
        M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        if M > self.N_sigma:
            k_r = (2 * self.N_sigma / M) - (self.N_sigma ** 2 / M ** 2)
            return k_r

        return 1

    def _compute_message_going_right(self):
        self.update_eta_lmbda()

        in_eta, in_lmbda = var_nodes[self.Et_id].eta, var_nodes[self.Et_id].lmbda
        k_r = self.huber_scale()

        # Compute the eta and lambda by var-factor rule and then scale
        eta_here, lambda_here = torch.clone(self.eta), torch.clone(self.lmbdap)

        eta_here[0] = self.eta[0] + in_eta
        lambda_here[0,0] = self.lmbdap[0,0] + in_lmbda

        eta_here *= k_r
        lambda_here *= k_r

        eta_b, eta_a = eta_here
        lambda_ba, lambda_bb = lambda_here[0]
        lambda_aa, lambda_ab = lambda_here[1]

        # Eq. 2.60 and 2.61 in Ortiz. 2003
        lambda_ab_lambda_bbinv = lambda_ab * 1/lambda_bb
        out_eta = torch.tensor([eta_a - (lambda_ab_lambda_bbinv * eta_b)])
        out_lmbda = torch.tensor([lambda_aa - (lambda_ab_lambda_bbinv * lambda_ba)])

        # Apply message damping
        if self.prev_msg_right is not None:
            out_eta = self.beta * out_eta + (1 - self.beta) * self.prev_msg_right[0]
            out_lmbda = self.beta * out_lmbda + (1 - self.beta) * self.prev_msg_right[1]

        self.out_eta = out_eta
        self.out_lmbda = out_lmbda

    def _compute_message_going_left(self):
        self.update_eta_lmbda()

        in_eta, in_lmbda = var_nodes[self.Etp_id].get_eta(), var_nodes[self.Etp_id].get_lmbda()
        k_r = self.huber_scale()

        # Compute the eta and lambda by var-factor rule and then scale
        eta_here, lambda_here = torch.clone(self.eta), torch.clone(self.lmbdap)

        eta_here[1] = self.eta[1] + in_eta
        lambda_here[1,1] = self.lmbdap[1,1] + in_lmbda

        eta_here *= k_r
        lambda_here *= k_r

        eta_a, eta_b = eta_here
        lambda_aa, lambda_ab = lambda_here[0]
        lambda_ba, lambda_bb = lambda_here[1]

        # Eq. 2.60 and 2.61 in Ortiz. 2003
        lambda_ab_lambda_bbinv = lambda_ab * 1/lambda_bb
        out_eta = torch.tensor([eta_a - (lambda_ab_lambda_bbinv * eta_b)])
        out_lmbda = torch.tensor([lambda_aa - (lambda_ab_lambda_bbinv * lambda_ba)])

        if self.prev_msg_left is not None:
            out_eta = self.beta * out_eta + (1 - self.beta) * self.prev_msg_left[0]
            out_lmbda = self.beta * out_lmbda + (1 - self.beta) * self.prev_msg_left[1]

        self.out_lmbda = out_lmbda
        self.out_eta = out_eta

    def compute_messages(self):
        self._compute_message_going_left()

    def __str__(self):
        return f'Dynamics Factor {self.f_id} connecting {self.Et_id} to {self.Etp_id}'

def print_fg(vars, factors):
    for i, j in zip(vars, factors):
        print(vars[i])
        print(factors[j])

def update_observational_factor(key):
    if not isinstance(key, tuple):
        factor_nodes[key].compute_messages()

def update_variable_belief(key):
    var_nodes[key].compute_messages()

def update_dynamics_factor(key):
    if isinstance(key, tuple):
        factor_nodes[key].compute_messages()

if __name__ == "__main__":
    sigma_smoothness = 1e7
    sigma_obs = 1e-4

    signal = np.load('E_synthetic_k5_Pdot2_noisy.npy')[:100].astype(np.float32)

    r = .2
    tau_E = 1.
    P = .2
    k = 5.

    # T = 1
    # dt = .01
    iters = 10

    t = np.arange(0, len(signal), 1)

    E0 = 0.2

    for iters in [10]:

        for i in range(len(t)):
            var_nodes[i] = Oscillator(i, torch.tensor([[E0]], requires_grad=True), torch.tensor([[0.0]], requires_grad=True), -1 if i == 0 else (i-1, i), -1 if i+1 == len(t) else (i, i+1), i)
            factor_nodes[i] = ObservationFactor(i, i, signal[i], torch.tensor([[sigma_obs ** -2]]))

        for i in range(len(t)):
            if i+1 < len(t):
                factor_nodes[(i, i+1)] = DynamicsFactor((i, i+1), torch.tensor([[sigma_smoothness ** -2]]), i, i+1, 0.1)

        # print_fg(var_nodes, factor_nodes)

        for i in range(iters):
            for key in factor_nodes:
                update_observational_factor(key)

            for key in var_nodes:
                update_variable_belief(key)

            for key in factor_nodes:
                update_dynamics_factor(key)

        # for _ in range(iters):
        #     # Sweep from left to right in time
        #     for i in range(len(t)):
        #         update_observational_factor(i)
        #         update_variable_belief(i)

        #         # Update dynamics factors
        #         if i+1 < len(t):
        #             update_dynamics_factor((i, i+1))

        recons_signal = torch.tensor([var_nodes[key].mu[0] for key in var_nodes])
        # x_vals = signal[:-1]
        # y_vals = signal[1:]

        # plt.scatter(x_vals, y_vals)

        print(recons_signal)
        plt.plot(recons_signal, label=rf'Iterations, iters = {iters}')

    plt.plot(signal, label='Noisy Signal')
    # plt.plot(clean_signal, label='Clean Signal')
    plt.legend()
    plt.show()
