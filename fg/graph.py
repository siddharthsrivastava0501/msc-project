from .gaussian import Gaussian
from typing import Any

class Graph:
    def __init__(self):
        self.var_nodes    = {}
        self.factor_nodes = {}
        self.param_ids    : list[Any] = []

    def get_var_belief(self, key) -> Gaussian:
        return self.var_nodes[key].belief

    def send_msg_to_factor(self, sender, recipient, msg : Gaussian) -> None:
        self.factor_nodes[recipient].inbox[sender] = msg

    def send_msg_to_variable(self, sender, recipient, msg : Gaussian) -> None:
        self.var_nodes[recipient].inbox[sender] = msg

    def update_dynamics_factor(self, key):
        if isinstance(key, tuple):
            self.factor_nodes[key].compute_and_send_messages()

    def update_params(self):
        for p in self.param_ids:
            self.var_nodes[p].compute_and_send_messages()

    def send_initial_parameter_messages(self):
        for p in self.param_ids:
            self.var_nodes[p].send_initial_messages()

    def update_all_observational_factors(self):
        for key in self.factor_nodes:
            if not isinstance(key, tuple):
                self.factor_nodes[key].compute_and_send_messages()

    def update_variable_belief(self, key):
        self.var_nodes[key].update_belief()

    def update_factor_belief(self, key):
        self.factor_nodes[key].update_belief()

    def update_all_beliefs(self):
        for key in self.var_nodes:
            self.update_variable_belief(key)
        for key in self.factor_nodes:
            self.update_factor_belief(key)
