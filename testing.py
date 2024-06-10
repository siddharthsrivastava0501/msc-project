import numpy as np

def sig(x):
    return 1/(1 + np.exp(-x))

r = .2
tau_E = 1.
tau_I = 2.
k = 5

P = .2
Q = .5

class VariableNode:
    def __init__(self, var_id, init_eta=0.0, init_lmbda=1.0):
        self.var_id = var_id
        self.eta = np.array([init_eta])
        self.lmbda = np.array([init_lmbda])

    def get_eta(self):
        return self.eta

    def get_lmbda(self):
        return self.lmbda

    def update(self, eta, lmbda):
        self.eta = eta
        self.lmbda = lmbda

    @property
    def value(self):
        return self.eta / self.lmbda

class DynamicsFactor:
    def __init__(self, f_id, lmbda_in, E_t_id, E_tp1_id, dt):
        self.f_id = f_id
        self.lmbda_in = lmbda_in
        self.E_t_id = E_t_id
        self.E_tp1_id = E_tp1_id
        self.dt = dt
        self.out_eta = {id: np.array([0.0]) for id in [E_t_id, E_tp1_id]}
        self.out_lmbda = {id: np.array([0.0]) for id in [E_t_id, E_tp1_id]}

    def dedt(t, E):
        de = (-E + sig(k*E + k*P)) / tau_E
        return np.array([de])
    
    def d_dedt_dE(self, E_t):
        # Derivative of Wilson-Cowan equation
        return (-1/tau_E) * sig(E_t) * (1 - sig(E_t))

    def compute_jacobian(self, E_t):
        j_a = 1 + self.dt * self.d_dedt_dE(E_t)
        j_b = -1
        return np.array([[j_a[0], j_b]])

    def update_message(self, var_nodes):
        E_t = var_nodes[self.E_t_id].value
        E_tp1 = var_nodes[self.E_tp1_id].value

        J = self.compute_jacobian(E_t)
        z = E_t + self.dt * self.dedt(E_t) - E_tp1

        eta = (J.T @ self.lmbda_in) * z
        lmbdap = (J.T @ self.lmbda_in) @ J

        # Compute messages for E_t and E_{t+1}
        for idx, var_id in enumerate([self.E_t_id, self.E_tp1_id]):
            in_eta, in_lmbda = var_nodes[var_id].get_eta(), var_nodes[var_id].get_lmbda()
            
            eta_here = np.copy(eta)
            lambda_here = np.copy(lmbdap)
            
            eta_here[idx] += in_eta
            lambda_here[idx, idx] += in_lmbda
            
            other_idx = 1 - idx  # if idx is 0, other_idx is 1, and vice versa
            
            self.out_eta[var_id] = eta_here[idx] - lambda_here[idx, other_idx] / lambda_here[other_idx, other_idx] * eta_here[other_idx]
            self.out_lmbda[var_id] = lambda_here[idx, idx] - lambda_here[idx, other_idx]**2 / lambda_here[other_idx, other_idx]

    def get_eta(self, var_id):
        return self.out_eta[var_id]

    def get_lmbda(self, var_id):
        return self.out_lmbda[var_id]

def run_belief_propagation(var_nodes, factors, max_iters=10):
    for _ in range(max_iters):
        for factor in factors:
            factor.update_message(var_nodes)
        
        for var_node in var_nodes.values():
            eta_sum = sum(factor.get_eta(var_node.var_id) for factor in factors if var_node.var_id in factor.out_eta)
            lmbda_sum = sum(factor.get_lmbda(var_node.var_id) for factor in factors if var_node.var_id in factor.out_lmbda)
            var_node.update(eta_sum, lmbda_sum)

# Example usage
var_nodes = {
    'E_0': VariableNode('E_0', init_eta=0.5, init_lmbda=1.0),
    'E_1': VariableNode('E_1'),
    'E_2': VariableNode('E_2'),
    'E_3': VariableNode('E_3')
}

factors = [
    DynamicsFactor('f_0_1', np.array([[1.0]]), 'E_0', 'E_1', dt=0.1),
    DynamicsFactor('f_1_2', np.array([[1.0]]), 'E_1', 'E_2', dt=0.1),
    DynamicsFactor('f_2_3', np.array([[1.0]]), 'E_2', 'E_3', dt=0.1)
]

run_belief_propagation(var_nodes, factors)

# Print results
for var_id, var_node in var_nodes.items():
    print(f"{var_id}: {var_node.value}")