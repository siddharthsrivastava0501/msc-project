import numpy as np

class LIFNetwork():

    def __init__(self, N, Dmax):
        self._Dmax = Dmax+1
        self.N = N
        self._X = np.zeros((Dmax + 1, N))
        self._I = np.zeros(N)
        self._cursor = 0
        self._lastFired = np.array([False]*N)
        self._dt = 0.1
        self.v = np.array([-65.0]*N)

    def set_delays(self, D):
        if D.shape != (self.N, self.N):
            raise Exception(f'Invalid shape for delay matrix, expected {(self.N, self.N)}, got {D.shape}')

        self._D = D
    
    def set_weights(self, W):
        if W.shape != (self.N, self.N):
            raise Exception(f'Invalid shape for weights matrix, expected {(self.N, self.N)}, got {W.shape}')
        
        self._W = W
    
    def set_current(self, I):
        if len(I) != self.N:
            raise Exception(f'Current vector must be of size {self.N}')

        self._I = I
        
    def set_params(self, Vr, R, tau):
        self.Vr = np.array(Vr)
        self.R = R
        self.tau = tau

        
    def update(self):
        """
        Simulate one millisecond of network activity. The internal dynamics
        of each neuron are simulated using the Euler method with step size
        self._dt, and spikes are delivered every millisecond.

        Returns the indices of the neurons that fired this millisecond.
        """

        self.v[self._lastFired] = self.Vr[self._lastFired]

        I = self._I + self._X[self._cursor%self._Dmax,:]

        fired = np.array([False]*self.N)
        for _ in range(int(1/self._dt)):
            notFired = np.logical_not(fired)
            v = self.v[notFired]
            self.v[notFired] += self._dt*((self.Vr - v + self.R*I[notFired])/self.tau)
            fired = np.logical_or(fired, self.v > -50)

        fired_idx = np.where(fired)[0]
        self._lastFired = fired
        self.v[fired] = 30*np.ones(len(fired_idx))

        self._I = np.zeros(self.N)

        for i in fired_idx:
            self._X[(self._cursor + self._D[i, :])%self._Dmax, range(self.N)] += self._W[i,:]

        self._X[self._cursor%self._Dmax,:] = np.zeros(self.N)
        self._cursor += 1

        return fired_idx
