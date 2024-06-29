from .gaussian import Gaussian

class Graph:
    def __init__(self):
        self.var_nodes = {}
        self.factor_nodes = {}

    def get_var_belief(self, key) -> Gaussian:
        return self.var_nodes[key].belief

    def get_factor_belief(self, key) -> Gaussian:
        return self.factor_nodes[key].belief

    def send_msg_to_factor(self, sender, recipient, msg : Gaussian) -> None:
        self.factor_nodes[recipient].inbox[sender] = msg

    def send_msg_to_variable(self, sender, recipient, msg : Gaussian) -> None:
        self.var_nodes[recipient].inbox[sender] = msg
