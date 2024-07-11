import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

'''
Lot of the code has been lifted from 
http://martinosorb.github.io/notebooks/2016/05/26/wilsoncowan.html#numerical
'''

def sigmoid(x, a, thr):
    return 1 / (1 + np.exp(-a * (x - thr)))

def Se(x):
    aE = 1.3
    thrE = 4
    return sigmoid(x, thrE, aE) - sigmoid(0, thrE, aE)

def Si(x):
    aI = 2
    thrI = 3.7
    return sigmoid(x, thrI, aI) - sigmoid(0, thrI, aI)


n_regions = 100

k1 = np.random.uniform(14, 18, n_regions)  
k2 = np.random.uniform(10, 14, n_regions)  
k3 = np.random.uniform(13, 17, n_regions)  
k4 = np.random.uniform(2, 4, n_regions)    

rE = 1.
rI = 1.

P = 1.
Q = 1

T = 100
dt = 0.1
time = np.arange(0, T, dt)


C = pd.read_csv('./fmri/DTI_fiber_consensus_HCP.csv', header=None).to_numpy()[:n_regions, :n_regions]
np.fill_diagonal(C, 0.)
C /= np.sum(C, axis=1)[:, np.newaxis]

E = np.zeros((len(time), n_regions))
I = np.zeros((len(time), n_regions))
E[0] = np.random.normal(0.2, 0.1)
I[0] = np.random.normal(0.2, 0.1)

for t in range(len(time)-1):
    # Calculate the input from other excitatory populations
    E_input = np.dot(C, E[t])
    
    for r in range(n_regions):
        E[t+1, r] = E[t, r] + dt * (-E[t, r] + (1 - rE * E[t, r]) * Se(k1[r] * E_input[r] - k2[r] * I[t, r] + P))
        I[t+1, r] = I[t, r] + dt * (-I[t, r] + (1 - rI * I[t, r]) * Si(k3[r] * E[t, r] - k4[r] * I[t, r] + Q))

# plot the solution in time for a few selected regions
plt.figure(figsize=(70, 70))

for i, region in enumerate(range(n_regions)):
    plt.subplot(10, 10, i+1)
    plt.plot(time, E[:, region] - I[:, region], label=f"E - I")
    # plt.plot(time, , label=f"I")
    plt.title(f"Region {region}")
    plt.xlabel("Time")
    plt.ylabel("Activity")
    plt.legend()

plt.tight_layout()
plt.show()