from fg.variables import Variable, Parameter
from fg.factors import PriorFactor
from torch.nn.functional import linear, relu
import torch
from fg.graph import Graph
from fg.gaussian import Gaussian
from collections import defaultdict
import matplotlib.pyplot as plt

class MLPFactor:
    def __init__(self, factor_id, z, lmbda_in, graph : Graph, huber = False):
        self.factor_id = factor_id

        self.lmbda_in = lmbda_in
        self.graph = graph
        self.huber = huber

        self._connected_vars = ['xt'] + [f'x{i}' for i in range(d)] + [f'p{i}' for i in range(d)] 

        self.z = z

        self.inbox = {}

        # Used for message damping, see Ortiz (2023) 3.4.6
        self._prev_messages = {}

    def _h_fn(self, Nt, *input):        
        X = torch.cat(input[:d]).T
        W = torch.cat(input[d:2*d]).T

        act = linear(X, W)

        return torch.abs(Nt - act)

    def linearise(self) -> Gaussian:
        '''
        Returns the linearised Gaussian factor based on equations 2.46 and 2.47 in Ortiz (2023)
        '''
        connected_variables = []
        for i in self._connected_vars:
            mean = self.graph.get_var_belief(i).mean.detach().clone()
            if mean.numel() > 1: #nD beliefs
                for j in range(mean.numel()):
                    connected_variables.append(mean[j].reshape(1, 1))
            else: #1D beliefs
                connected_variables.append(mean.reshape(1, 1))

        Nt = connected_variables[0]
        weights = connected_variables[1:]

        self.h = self._h_fn(Nt, *weights)

        J = torch.concat(torch.autograd.functional.jacobian(self._h_fn, (Nt, *weights)), 0)[..., 0, 0].T
        x0 = torch.concat([v for v in connected_variables], dim=0)

        eta = (J.T @ self.lmbda_in) @ (-self.h.T + J @ x0)
        lmbda = (J.T @ self.lmbda_in) @ J 

        return Gaussian.from_canonical(eta.detach(), lmbda.detach())

    def _compute_message_to_i(self, i, beta = 0.5) -> Gaussian:
        '''
        Compute message to variable at index i in `self._vars`,
        All of this is eqn 8 from 'Learning in Deep Factor Graphs with Gaussian Belief Propagation'
        '''
        linearised_factor = self.linearise()

        product = Gaussian.zeros_like(linearised_factor)

        # Build our message product by adding corresponding eta and lambda
        # in product
        k = 0
        for j, id in enumerate(self._connected_vars):
            if j != i:
                in_msg = self.inbox.get(id, Gaussian.from_canonical(torch.tensor([0.]), \
                    torch.tensor([0.])))

                offset = in_msg.eta.numel()
                product.eta[k : k+offset] += in_msg.eta
                product.lmbda[k : k+offset, k : k+offset] += in_msg.lmbda

                k += offset
            else:
                k += self.graph.var_nodes[self._connected_vars[i]].num_vars
                # k += 2 if i in [0,1] else 1

        factor_product = linearised_factor * product

        start_idx = 0
        for k in range(i):
            start_idx += self.graph.var_nodes[self._connected_vars[k]].num_vars

        idx_to_marginalise = list(range(start_idx, start_idx + self.graph.var_nodes[self._connected_vars[i]].num_vars))

        marginal = factor_product.marginalise(idx_to_marginalise)

        kR = self.compute_huber() if self.huber else 1.
        marginal *= kR

        prev_msg = self._prev_messages.get(i, Gaussian.zeros_like(marginal))
        damped_factor = (marginal * beta) * (prev_msg * (1 - beta))

        # Store previous message
        self._prev_messages[i] = damped_factor

        return damped_factor

    def compute_and_send_messages(self) -> None:
        for i, var_id in enumerate(self._connected_vars):
            msg = self._compute_message_to_i(i)
            self.graph.send_msg_to_variable(self.factor_id, var_id, msg)

    def __str__(self):
        return f'MLP: Var: {self.var_id}' 


# Set up problem
n = 150
d = 6
start_idx = 0

gt = torch.rand(1, d)*10
X = torch.rand(n, d)
y = linear(X, gt)

factor_graph = Graph(1)

factor_graph.factor_nodes['mlp'] = MLPFactor(
    factor_id = 'mlp',
    lmbda_in = torch.tensor([[0.1 ** -1.]]),
    graph = factor_graph,
    z = torch.tensor([[0.]])
)

factor_graph.var_nodes['xt'] = Variable(
    id = 'xt',
    belief = Gaussian(torch.tensor([[0.]]), torch.tensor([[0.2]])),
    num_vars = 1,
    graph = factor_graph,
    connected_factors = ['mlp']
)

factor_graph.factor_nodes['xt_prior'] = PriorFactor(
    factor_id = 'xt_prior',
    var_id = 'xt',
    z = torch.tensor([[y[start_idx]]]),
    lmbda_in = torch.tensor([[0.1 ** -2.]]),
    graph = factor_graph
)

for i in range(d):
    factor_graph.var_nodes[f'x{i}'] = Variable(
        id = f'x{i}',
        belief = Gaussian(torch.tensor([[0.]]).T, torch.tensor([[1e-1]])),
        num_vars = 1,
        connected_factors = ['mlp'],
        graph = factor_graph
    )

    factor_graph.factor_nodes[f'x{i}_obs'] = PriorFactor(
        factor_id = f'x{i}_obs',
        var_id = f'x{i}',
        z = torch.tensor([[X[start_idx,i].item()]]),
        lmbda_in = torch.tensor([[1e-1 ** -2]]),
        graph = factor_graph
    )

    factor_graph.param_ids.append(f'p{i}')
    factor_graph.var_nodes[f'p{i}'] = Parameter(
        id     = f'p{i}', 
        belief = Gaussian(torch.tensor([[0.]]), torch.tensor([[10 ** 2.]])),
        graph  = factor_graph,
        connected_factors = ['mlp']
    )

    # Add priors to those parameters
    factor_graph.factor_nodes[f'p{i}_prior'] = PriorFactor(
        factor_id = f'p{i}_prior',
        var_id = f'p{i}',
        z = torch.tensor([[0.]]).T, 
        lmbda_in = torch.diag(torch.tensor([10 ** -2.])),
        graph = factor_graph
    )

# Start GBP
factor_graph.update_all_observational_factors()

print(gt)

res_mean = defaultdict(lambda: [])
res_cov = defaultdict(lambda: [])

for i in range(n):
    for j in range(d):
        factor_graph.factor_nodes[f'x{j}_obs'].set_z(torch.tensor([[X[i,j].item()]]))
    factor_graph.factor_nodes['xt_prior'].set_z(torch.tensor([[y[i].item()]]))

    factor_graph.update_all_observational_factors()

    for _ in range(100):
        for _, j in factor_graph.var_nodes.items(): j.compute_and_send_messages()

        factor_graph.factor_nodes['mlp'].compute_and_send_messages()

        for _, j in factor_graph.var_nodes.items(): j.update_belief()
    
    # Use the param belief as new priors
    for j in range(d):
        factor_graph.factor_nodes[f'p{j}_prior'].belief = factor_graph.var_nodes[f'p{j}'].belief

    print(f'--- {i} ---')
    for j in range(d):
        res_mean[j].append(factor_graph.var_nodes[f'p{j}'].mean.item())
        res_cov[j].append(factor_graph.var_nodes[f'p{j}'].cov.item())
        print(factor_graph.var_nodes[f'p{j}'].mean, factor_graph.var_nodes[f'p{j}'].cov)