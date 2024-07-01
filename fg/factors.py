from collections import defaultdict
import torch
from torch._prims_common import Tensor
from .graph import Graph
from .gaussian import Gaussian
from .simulation_config import sig, dEdt

class ObservationFactor:
    def __init__(self, factor_id, var_id, z, lmbda_in, graph : Graph) -> None:
        self.factor_id = factor_id
        self.var_id = var_id
        self.z = z
        self.lmbda_in = lmbda_in

        J = torch.tensor([[1.]])

        # Equation 2.46, 2.47 in Ortiz (2003)
        self.belief = Gaussian.from_canonical((J.T @ lmbda_in) * z, (J.T @ lmbda_in) @ J)

        # Huber threshold
        self.N_sigma = torch.sqrt(self.lmbda_in[0,0])

        self.inbox = defaultdict(lambda: Gaussian.zeros_like(self.belief))

        self.graph = graph

    
    def compute_huber(self) -> float:
        # Equation 3.16 in Ortiz (2023)
        r = self.z - self.graph.get_var_belief(self.var_id).mean
        M = torch.sqrt(r * self.lmbda_in * r)

        # Equation 3.20 in Ortiz (2023)
        if M > self.N_sigma:
            kR = (2 * self.N_sigma / M) - (self.N_sigma**2 / M**2)
            return kR.item()

        return 1.

    def compute_and_send_messages(self) -> None:
        kR = self.compute_huber()

        self.graph.send_msg_to_variable(self.factor_id, self.var_id, self.belief * kR)

class DynamicsFactor:
    '''
    Represents a dynamics factor that enforces dynamics between `Et_id` (left) and `Etp_id` (right),
    and is also connected to learnable parameters given by `parameters`.
    '''
    def __init__(self, Et_id, Etp_id, lmbda_in : Tensor, factor_id, graph : Graph) -> None:
        self.Et_id, self.Etp_id = Et_id, Etp_id
        self.lmbda_in = lmbda_in
        self.factor_id = factor_id
        self.graph : Graph = graph

        self.parameters = graph.param_ids
        self.belief = Gaussian.from_canonical(torch.tensor([[0.]]), torch.tensor([[0.]]))
        self.N_sigma = torch.sqrt(lmbda_in)
        self.z = 0

        self.inbox = defaultdict(lambda: Gaussian.zeros_like(self.belief))

        # Used for message damping, see Ortiz (2023) 3.4.6
        self._prev_messages = defaultdict(lambda: Gaussian.zeros_like(self.belief))

        self._connected_vars = [Et_id, Etp_id] + list(self.parameters)


    def linearise(self) -> None:
        '''
        Returns the linearised Gaussian factor based on equations 2.46 and 2.47 in Ortiz (2023)
        '''
        # This ugly line extracts the means of all the beliefs of our adj. parameters and gets them ready for autograd
        connected_variables = [self.graph.get_var_belief(i).mean.detach().clone().requires_grad_(True) \
            for i in self._connected_vars]

        Et_mu, Etp_mu = connected_variables[0], connected_variables[1]

        # Measurement function h = |Etp - (Et + deltaT * dedt)|, want to minimise h to minimise energy
        self.h = torch.abs(Etp_mu - (Et_mu + 0.01 * dEdt(Et_mu, *connected_variables[2:])))
        self.h.backward()

        J = torch.tensor([[v.grad for v in connected_variables]])
        x0 = torch.tensor([v.item() for v in connected_variables])

        eta = J.T @ self.lmbda_in @ (J @ x0 - self.h)
        lmbda = (J.T @ self.lmbda_in) @ J

        self.belief = Gaussian.from_canonical(eta, lmbda)

    def compute_huber(self) -> float:
        # Equation 3.16 in Ortiz (2023)
        r = self.z - self.h
        M = torch.sqrt(r * self.lmbda_in * r)

        # Equation 3.20 in Ortiz (2023)
        if M > self.N_sigma:
            kR = (2 * self.N_sigma / M) - (self.N_sigma**2 / M**2)
            return kR.item()

        return 1.

    def _compute_message_to_i(self, i, beta = 0.8) -> Gaussian:
        '''
        Compute message to variable at index i in `self._vars`,
        All of this is eqn 8 from 'Learning in Deep Factor Graphs with Gaussian Belief Propagation'
        '''
        self.linearise()

        factor = self.belief.clone()

        for j,msg in self.inbox.items():
            if j == i: continue

            factor *= msg

        kR = self.compute_huber()
        factor *= kR

        factor = factor.marginalise(i)

        damped_factor = (factor * beta) * (1 - beta) * (self._prev_messages[i])

        # Store previous message
        self._prev_messages[i] = factor

        print('marginalisation result', factor)
        print('sending message', damped_factor)

        return damped_factor

    def compute_and_send_messages(self) -> None: pass

    def compute_messages_except_key(self, key = None):
        '''
        Sends messages to all adjacent variables except that of `key`. For example, if we wish to send messages right,
        during a right sweep, we would send messages to all vars. except Et.
        '''
        for i,j in enumerate(self._connected_vars):
            if j == key: continue

            self.graph.send_msg_to_variable(self.factor_id, j, self._compute_message_to_i(i))
