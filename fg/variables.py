import torch
from collections import defaultdict

from torch.linalg import Tensor
from .gaussian import Gaussian
from .graph import Graph

class Variable:
    def __init__(self, var_id, belief : Gaussian, left_id, right_id, prior_id, graph : Graph) -> None:
        self.var_id = var_id
        self.belief = belief

        self.left_id = left_id
        self.right_id = right_id
        self.prior_id = prior_id

        self.inbox = defaultdict(lambda: Gaussian.zeros_like(self.belief))
        self.connected_factors = [prior_id, left_id, right_id]

        self.graph = graph

    @property
    def mean(self) -> Tensor:
        return self.belief.mean

    @property
    def cov(self) -> Tensor:
        return self.belief.cov

    @property
    def eta(self) -> Tensor:
        return self.belief.eta

    @property
    def lmbda(self) -> Tensor:
        return self.belief.lmbda

    def belief_update(self) -> None:
        '''
        Consume the messages in the inbox to update belief
        '''
        curr = Gaussian.zeros_like(self.belief)

        for _, message in self.inbox.items():
            curr *= message

        self.belief = curr

    def compute_and_send_messages(self) -> None:
        '''
        Equation 2.50, 2.51 in Ortiz (2023)
        '''
        self.belief_update()
        for fid in self.connected_factors:
            if fid == -1: continue

            msg = self.belief / self.graph.get_factor_belief(fid)

            self.graph.send_msg_to_factor(self.var_id, fid, msg)

        self.inbox.clear()

    def __str__(self):
        return f'Variable {self.var_id} [mu={self.mean}, cov={self.cov}]'

class Parameter:
    def __init__(self, param_id, belief : Gaussian, graph : Graph, connected_factors : list):
        self.param_id = param_id
        self.belief = belief

        self.inbox = defaultdict(lambda: Gaussian.zeros_like(self.belief))
        self.connected_factors = connected_factors
        self.graph = graph

    @property
    def mean(self) -> Tensor:
        return self.belief.mean

    @property
    def cov(self) -> Tensor:
        return self.belief.cov

    @property
    def eta(self) -> Tensor:
        return self.belief.eta

    @property
    def lmbda(self) -> Tensor:
        return self.belief.lmbda

    def belief_update(self) -> None:
        '''
        Consume the messages in the inbox to update belief
        '''
        curr = Gaussian.zeros_like(self.belief)

        for _, message in self.inbox.items():
            curr *= message

        self.belief = curr

    def compute_and_send_messages(self) -> None:
        self.belief_update()
        for fid in self.connected_factors:
            if fid == -1: continue

            msg = self.belief / self.graph.get_factor_belief(fid)

            self.graph.send_msg_to_factor(self.param_id, fid, msg)

        self.inbox.clear()

    def __str__(self):
        return f'Variable {self.param_id} [mu={self.mean}, cov={self.cov}]'
