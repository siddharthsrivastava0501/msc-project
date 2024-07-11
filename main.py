from fg.variables import Variable, Parameter
from fg.factors import DynamicsFactor, ObservationFactor, PriorFactor
from fg.functions import sig
from fg.simulation_config import simulate_signal
from fg.graph import Graph
from fg.gaussian import Gaussian
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sigma_obs = 1e-2
    sigma_dynamics = 1e-3
    sigma_prior = 1e1
    iters = 200

    config = {
        'T': 6.,
        'dt': 0.01,
        'k1': 5.,
        'k2': 5.,
        'k3': 5.,
        'k4': 5.,
        'P': 0.2,
        'Q': 0.2,
    }

    E, I = simulate_signal(config)
    t = torch.arange(0, len(E), 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    factor_graph = Graph()

    # Add 0 as id as we populate them later
    param_dict = {
        'k1': Parameter(0, Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])), factor_graph, []),
        'k2': Parameter(0, Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])), factor_graph, []),
        'k3': Parameter(0, Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])), factor_graph, []),
        'k4': Parameter(0, Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])), factor_graph, []),
        'P':  Parameter(0, Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])), factor_graph, []),
        'Q':  Parameter(0, Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])), factor_graph, [])
    }

    # -- Construct FG -- #
    # Add our variable and observation factors at each time step
    for i in range(len(t)):
        factor_graph.var_nodes[i] = Variable(i, Gaussian(torch.tensor([[0., 0.]]).T, torch.tensor([[0.2, 0.], [0., 0.2]])), -1 if i == 0 else (i-1, i), -1 if i+1 == len(t) else (i,i+1), i, factor_graph)
        factor_graph.factor_nodes[i] = ObservationFactor(i, i, torch.tensor([[E[i], I[i]]]).T, torch.tensor([[sigma_obs ** -2, 0.], [0., sigma_obs ** -2]]), factor_graph)

    # Add our parameters as additional variables to our factor graph
    k = len(t)
    for _,p in param_dict.items():
        p.id = k
        factor_graph.param_ids.append(p.id)
        factor_graph.var_nodes[p.id] = p

        k += 1

    # Connect dynamics factors between timestep i and i+1 and connect each dyn. factor to our parameters
    for i in range(len(t)):
        if i+1 < len(t):
            dyn_id = (i, i+1)
            factor_graph.factor_nodes[dyn_id] = DynamicsFactor(i, i+1, torch.tensor([[sigma_dynamics ** -2, 0.], [0., sigma_dynamics ** -2]]), dyn_id, factor_graph)

            for _,p in param_dict.items():
                p.connected_factors.append(dyn_id)

    # Zero mean priors on the parameters
    for i, (_,p) in enumerate(param_dict.items()):
        factor_graph.factor_nodes[i + p.id] = PriorFactor(i + p.id, p.id, torch.tensor([0.]),
            torch.diag(torch.tensor([sigma_prior ** 2.])), factor_graph)

    # == RUN GBP (Sweep schedule) === #
    for iter in range(iters):
        print(f'Iteration {iter}')
        for k, v in param_dict.items():
            print(k, v)
        print('------')

        if iter == 0:
            # Initialise messages from observation factors to variables
            # and prior factors to parameters (if learning params)
            factor_graph.update_all_observational_factors()

            # Now update messages from variables to factors
            # This should ensure all var to dynamics factor messages have non-zero precision
            for i in factor_graph.var_nodes:
                curr = factor_graph.var_nodes[i]
                curr.compute_and_send_messages()

            factor_graph.prune()

        # -- RIGHT PASS --
        for i in range(len(t)):
            curr = factor_graph.var_nodes[i]
            curr.compute_and_send_messages()

            if curr.right_id == -1: continue

            fac = factor_graph.factor_nodes[curr.right_id]
            fac.compute_and_send_messages()

        factor_graph.update_params()

        # -- LEFT PASS --
        for i in range(len(t)-1, -1, -1):
            curr = factor_graph.var_nodes[i]
            curr.compute_and_send_messages()

            if curr.left_id == -1: continue

            fac = factor_graph.factor_nodes[curr.left_id]
            fac.compute_and_send_messages()

        factor_graph.update_params()

        factor_graph.update_all_beliefs()

    # Update params with what we have learnt
    for k, p in param_dict.items():
        config[k] = p.mean.item()

    E_rec, I_rec = simulate_signal(config)

    plt.plot(E, label='GT E')
    plt.plot(I, label='GT I')
    plt.plot(E_rec.detach().numpy(), label='E_rec')
    plt.plot(I_rec.detach().numpy(), label='I_rec')
    plt.legend()
    plt.show()
