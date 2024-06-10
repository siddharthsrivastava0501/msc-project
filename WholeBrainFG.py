import numpy as np
import matplotlib.pyplot as plt

signal = np.load('E_synthetic_k5_Pdot2_with_inhibitory.npy')

r = .2
tau_E = 1.
P = .2
k = 60

T = 10
dt = .01

t = np.arange(0, T + dt, dt)

def sig(x):
    return 1/(1 + np.exp(-x))

def dedt(E):
    # de = (-E + (1 - r*E)*sig(w_ee*E - w_ei*I + P)) / tau_E
    de = (-E + sig(k*E + k*P)) / tau_E
    return de

def d_dedt(E):
    s =  sig(k*E + k*P)
    dde = (-1 / tau_E) * s * (1 - s)
    
    return dde

var_nodes, factor_nodes = {}, {}

class Variable:
    def __init__(self, var_id, mu, sigma, left_dynamics_id, right_dynamics_id, obs_id):
        self.var_id = var_id
        self.mu = mu
        self.sigma = sigma
        self.factor_ids = [obs_id, left_dynamics_id, right_dynamics_id]
        self.eta = np.zeros_like(mu)
        self.lmbda = np.zeros_like(sigma)

    def get_mu(self):
        return self.mu
    
    def get_sigma(self):
        return self.sigma
    
    def get_eta(self):
        return self.eta
    
    def get_lmbda(self):
        return self.lmbda

    def belief_update(self):
        valid_factors = [fid for fid in self.factor_ids if fid != -1]
        
        self.eta = np.sum([factor_nodes[fid].get_eta() for fid in valid_factors], axis=0)
        self.lmbda = np.sum([factor_nodes[fid].get_lmbda() for fid in valid_factors], axis=0)

        self.sigma = np.linalg.inv(self.lmbda)
        self.mu = self.sigma @ self.eta

    def update_message(self):
        self.belief_update()
        up = self.factor_ids[1]

        if up == -1:
            self.eta, self.lmbda = np.zeros_like(self.eta), np.zeros_like(self.lmbda)
        else:
            in_eta, in_lambda = factor_nodes[up].get_eta(), factor_nodes[up].get_lmbda()

            self.eta -= in_eta
            self.lmbda -= in_lambda

    def __str__(self):
        return f'Variable {self.var_id} connected to {self.factor_ids}, with mu = {self.mu} and sigma = {self.sigma}'

class ObservationFactor:
    def __init__(self, f_id, var_id, z, lmbda_in):
        self.f_id = f_id
        self.z = z
        self.lmbda_in = lmbda_in
        self.var_id = var_id

        # Jacobian
        J = np.array([[1.]])

        self.eta = (J.T @ lmbda_in) * z
        self.lmbda = (J.T @ lmbda_in) @ J
        self.N_sigma = np.sqrt(lmbda_in[0,0])

        self.var_eta = self.eta
        self.var_lmbda = self.lmbda

    def get_eta(self):
        return self.var_eta
    
    def get_lmbda(self):
        return self.var_lmbda
    
    def huber_scale(self):
        r = self.z - var_nodes[self.var_id].mu
        M = np.sqrt(r * self.lmbda_in[0,0] * r)

        if M > self.N_sigma:
            kR = (2 * self.N_sigma / M) - (self.N_sigma ** 2 / M ** 2)
            return kR
        
        return 1

    def update_message(self):
        kR = self.huber_scale()
        self.var_eta = self.eta * kR
        self.var_lmbda = self.lmbda * kR

    # This derivation can be found in ch. 3.3
    # 'Ortiz, J. (2023). Gaussian Belief Propagation for Real-Time Decentralised Inference. Phd Thesis.'
    def huber_scale(self):
        r = self.z - var_nodes[self.var_id].get_mu()[0,0]
        M = np.sqrt(r * self.lmbda_in[0,0] * r)

        if M > self.N_sigma:
            kR = (2 * self.N_sigma / M) - (self.N_sigma ** 2 / M ** 2)
            return kR
        
        return 1

    def update_message(self):
        kR = self.huber_scale()
        self.var_eta = self.eta * kR
        self.var_lmbda = self.lmbda * kR

    def __str__(self):
        return f'Observation factor with connected to {self.var_id} and z = {self.z} '

class DynamicsFactor:
    ''' Et_id is the id of E_t, Etp_id is the id of E_{t+1}'''
    def __init__(self, f_id, lmbda_in, Et_id, Etp_id):
        self.Et_id = Et_id
        self.Etp_id = Etp_id
        self.f_id = f_id
        self.lmbda_in = lmbda_in
        self.out_eta = np.array([[0.0]])
        self.out_lmbda = np.array([[0.0]])

        self.E = var_nodes[Et_id].mu[0,0]

        J = np.array([[1 + dt * d_dedt(self.E), -1]])
        z = 0
        self.eta = (J.T @ lmbda_in) * z
        self.lmbdap = (J.T @ lmbda_in) @ J

        self.N_sigma = np.sqrt(lmbda_in[0,0])

    def get_eta(self): 
        return self.out_eta

    def get_lmbda(self):
        return self.out_lmbda
    
    def huber_scale(self):
        r = 0 - (var_nodes[self.Etp_id].mu - var_nodes[self.Et_id].mu)
        M = np.sqrt(r * self.lmbda_in[0,0] * r)

        if M > self.N_sigma:
            k_r = (2 * self.N_sigma / M) - (self.N_sigma ** 2 / M ** 2)
            return k_r
        
        return 1
    
    def update_message(self):
        in_eta, in_lmbda = var_nodes[self.Et_id].eta, var_nodes[self.Et_id].lmbda
        k_r = self.huber_scale()

        # Compute the eta and lambda by var-factor rule and then scale
        eta_here, lambda_here = np.copy(self.eta), np.copy(self.lmbdap)

        eta_here[1] = self.eta[1] + in_eta
        lambda_here[1,1] = self.lmbdap[1,1] + in_lmbda

        eta_here *= k_r
        lambda_here *= k_r

        eta_a, eta_b = eta_here
        lambda_aa, lambda_ab = lambda_here[0]
        lambda_ba, lambda_bb = lambda_here[1]

        # Eq. 2.60 and 2.61 in Ortiz. 2003
        lambda_ab_lambda_bbinv = lambda_ab * 1/lambda_bb
        self.out_eta = np.array([eta_a - (lambda_ab_lambda_bbinv * eta_b)])
        self.out_lmbda = np.array([lambda_aa - (lambda_ab_lambda_bbinv * lambda_ba)])

    def __str__(self):
        return f'Dynamics Factor {self.f_id} connecting {self.Et_id} to {self.Etp_id}'

def print_fg(vars, factors):
    for i, j in zip(vars, factors):
        print(vars[i])
        print(factors[j])

def update_observational_factor(key):
    if not isinstance(key, tuple):
        factor_nodes[key].update_message()

def update_variable_belief(key):
    var_nodes[key].update_message()

def update_dynamics_factor(key):
    if isinstance(key, tuple):
        factor_nodes[key].update_message()

if __name__ == "__main__":
    sigma_smoothness = 0.01
    sigma_obs = 2

    for i in range(len(t)):
        var_nodes[i] = Variable(i, np.array([[0.]]), np.array([[0.]]), -1 if i == 0 else (i-1, i), -1 if i+1 == len(t) else (i, i+1), i)
        factor_nodes[i] = ObservationFactor(i, i, signal[i], np.array([[sigma_obs ** -2]]))

        if i+1 < len(t):
            factor_nodes[(i, i+1)] = DynamicsFactor((i, i+1), np.array([[sigma_smoothness ** -2]]), i, i+1)

    # print_fg(factor_nodes, var_nodes)

    for key in factor_nodes:
        update_observational_factor(key)
    
    for key in var_nodes:
        update_variable_belief(key)
    
    for key in factor_nodes:
        update_dynamics_factor(key)


    recons_signal = np.array([var_nodes[key].mu[0] for key in var_nodes])
    plt.plot(signal)
    plt.plot(recons_signal)
    plt.show()
        