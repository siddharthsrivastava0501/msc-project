import torch
from torch import Tensor
from .graph import Graph
from .gaussian import Gaussian
from .simulation_config import dIdt, sig, dEdt

class ObservationFactor:
    def __init__(self, factor_id, var_id, z, lmbda_in, graph : Graph, huber = False) -> None:
        self.factor_id = factor_id
        self.var_id = var_id

        self.z = z
        self.Et_z, self.It_z = self.z

        self.lmbda_in = lmbda_in

        J = torch.eye(2)

        # Equation 2.46, 2.47 in Ortiz (2023)
        self.belief = Gaussian.from_canonical((J.T @ lmbda_in) @ z, (J.T @ lmbda_in) @ J)

        # Huber threshold
        self.N_sigma = torch.sqrt(self.lmbda_in[0,0])

        self.inbox = {}

        self.graph = graph

        self.huber = huber

    def update_belief(self) -> None: pass

    def compute_huber(self) -> float:
        # Equation 3.16 in Ortiz (2023)
        r = (self.Et_z - self.graph.get_var_belief(self.var_id).mean[0]) + \
            (self.It_z - self.graph.get_var_belief(self.var_id).mean[1])
        M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        # Equation 3.20 in Ortiz (2023)
        if M > self.N_sigma and self.huber:
            kR = (2 * self.N_sigma / M) - (self.N_sigma**2 / M**2)
            kR = kR.item()
        else:
            kR = 1.

        return kR

    def compute_and_send_messages(self) -> None:
        kR = self.compute_huber()

        message = self.belief * kR
        self.graph.send_msg_to_variable(self.factor_id, self.var_id, message)

    def __str__(self) -> str:
        return f'Obs: [{self.factor_id} -- {self.var_id}], z = {self.z}'


class PriorFactor:
    def __init__(self, factor_id, var_id, z, lmbda_in, graph : Graph, huber = False) -> None:
        self.factor_id = factor_id
        self.var_id = var_id

        self.z = z
        self.lmbda_in = lmbda_in

        J = torch.ones((1,1))

        # Equation 2.46, 2.47 in Ortiz (2023)
        self.belief = Gaussian.from_canonical((J.T @ lmbda_in) @ z, (J.T @ lmbda_in) @ J)

        # Huber threshold
        self.N_sigma = torch.sqrt(self.lmbda_in[0,0])

        self.inbox = {}

        self.graph = graph

        self.huber = huber

    def update_belief(self) -> None: pass

    def compute_huber(self) -> float:
        # Equation 3.16 in Ortiz (2023)
        r = self.z - self.graph.get_var_belief(self.var_id).mean
        M = torch.sqrt(r * self.lmbda_in[0,0] * r)

        # Equation 3.20 in Ortiz (2023)
        if self.huber:
            kR = (2 * self.N_sigma / M) - (self.N_sigma**2 / M**2)
            kR = kR.item()
        else:
            kR = 1.

        return kR

    def compute_and_send_messages(self) -> None:
        kR = self.compute_huber()

        message = self.belief * kR
        self.graph.send_msg_to_variable(self.factor_id, self.var_id, message)

    def __str__(self) -> str:
        return f'Prior: [{self.factor_id} -- {self.var_id}], z = {self.z}'


class DynamicsFactor:
    '''
    Represents a dynamics factor that enforces dynamics between `Et_id` (left) and `Etp_id` (right),
    and is also connected to learnable parameters given by `parameters`.
    '''
    def __init__(self, Vt_id, Vtp_id, lmbda_in : Tensor, factor_id, graph : Graph, huber = False) -> None:
        self.Vt_id, self.Vtp_id = Vt_id, Vtp_id
        self.lmbda_in = lmbda_in
        self.factor_id = factor_id
        self.graph : Graph = graph

        self.parameters = graph.param_ids

        self.N_sigma = torch.sqrt(lmbda_in)
        self.z = 0

        self.inbox = {}

        # Used for message damping, see Ortiz (2023) 3.4.6
        self._prev_messages = {}

        self._connected_vars = [Vt_id, Vtp_id] + list(self.parameters)

        self.huber = huber

    def linearise(self) -> Gaussian:
        '''
        Returns the linearised Gaussian factor based on equations 2.46 and 2.47 in Ortiz (2023)
        '''

        # Extracts the means of all the beliefs of our adj.
        # parameters and gets them ready for autograd
        connected_variables = []
        for i in self._connected_vars:
            mean = self.graph.get_var_belief(i).mean.detach().clone()
            if mean.numel() > 1: #nD beliefs
                for j in range(mean.numel()):
                    connected_variables.append(mean[j].reshape(1, 1).requires_grad_(True))
            else: #1D beliefs
                connected_variables.append(mean.reshape(1, 1).requires_grad_(True))

        Et_mu, It_mu = connected_variables[0:2]
        Etp_mu, Itp_mu = connected_variables[2:4]
        k1, k2, k3, k4 = connected_variables[4:8]
        P,Q = connected_variables[8:]

        h_ext = Etp_mu - (Et_mu + 0.01 * dEdt(Et_mu, It_mu, k1, k2, P))
        h_inh = Itp_mu - (It_mu + 0.01 * dIdt(Et_mu, It_mu, k3, k4, Q))

        # Measurement function h = Etp - (Et + deltaT * dEdt) + Itp - (It + deltaT * dIdt)
        # Want to minimise the Euler expansion of both the ext. DE and inh. DE
        self.h = h_ext + h_inh
        self.h.backward()

        J = torch.tensor([[v.grad for v in connected_variables if v.grad is not None]])
        x0 = torch.cat([v for v in connected_variables])

        eta = J.T @ self.lmbda_in @ (J @ x0 - self.h)
        lmbda = (J.T @ self.lmbda_in) @ J

        return Gaussian.from_canonical(eta.detach(), lmbda.detach())

    def compute_huber(self) -> float:
        # Equation 3.16 in Ortiz (2023)
        r = self.z - self.h
        M = torch.sqrt(r @ self.lmbda_in @ r)

        # Equation 3.20 in Ortiz (2023)
        if M > self.N_sigma and self.huber:
            kR = (2 * self.N_sigma / M) - (self.N_sigma**2 / M**2)
            kR = kR.item()
        else:
            kR = 1.

        return kR


    def _compute_message_to_i(self, i, beta = 0.3) -> Gaussian:
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

                # Element 0 and 1 in self._connected_vars will be the
                # EI oscillator vars, and they each have a 2D Gaussian as their belief
                # since they encode Et, It and Etp, Itp respectively. Therefore,
                # we have to correctly offset our product Gaussian with 2 if
                # our j is at the 0th or 1st element. Otherwise just continue as
                # normal.
                offset = 2 if j in [0,1] else 1
                product.eta[k : k+offset] += in_msg.eta
                product.lmbda[k : k+offset, k : k+offset] += in_msg.lmbda

                k += offset
            else:
                k += 2 if i in [0,1] else 1

        factor_product = linearised_factor * product

        # Absolute monkey logic at the moment to find out what indices to keep
        # when we perform the marginalisation for msg. passing
        match i:
            case 0: idx_to_marginalise = [0,1]      # This is Vt
            case 1: idx_to_marginalise = [2,3]      # This is Vtp
            case _: idx_to_marginalise = [i+2]      # This is the parameters

        marginal = factor_product.marginalise(idx_to_marginalise)

        kR = self.compute_huber()
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
