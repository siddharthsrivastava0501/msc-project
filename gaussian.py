import torch

class Gaussian:
    '''
    Defines a Gaussian with an eta and a Lambda
    '''

    def __init__(self, eta, lam):
        self.eta = torch.as_tensor(eta)
        self.lam = torch.as_tensor(lam)

    def get_cov(self):
        return torch.linalg.inv(self.lam)
    
    def get_mean(self):
        return self.get_cov() @ self.eta

    def prod(self, g2):
        self.eta += g2.eta
        self.lam += g2.lam

