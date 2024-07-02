from fg.variables import Variable, Parameter
from fg.factors import DynamicsFactor, ObservationFactor
from fg.simulation_config import simulate_signal
from fg.graph import Graph
from fg.gaussian import Gaussian
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sigma_obs = 1e-2
    sigma_prior = 1e1
    sigma_dynamics = 1e-3
    GT_k = 1.2
    T, dt = 15, 0.01
    iters = 20

    signal = simulate_signal(T, dt, GT_k)
    noise = torch.normal(0, sigma_obs, signal.shape)
    signal += noise
    # signal = [1., 2., 3., 4.]
    t = torch.arange(0, len(signal), 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    factor_graph = Graph()

    param_dict = {
        'k': Parameter(len(t), Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])), factor_graph, [])
    }

    # -- Construct FG -- #
    # Add our variable and observation factors at each time step
    for i in range(len(t)):
        factor_graph.var_nodes[i] = Variable(i, Gaussian(torch.tensor([[0.]]), torch.tensor([[0.2]])), -1 if i == 0 else (i-1, i), -1 if i+1 == len(t) else (i,i+1), i, factor_graph)
        factor_graph.factor_nodes[i] = ObservationFactor(i, i, signal[i], torch.tensor([[sigma_obs ** -2]]), factor_graph)

    # Add our parmaters as additional variables to our factor graph
    for _,p in param_dict.items():
        factor_graph.param_ids.append(p.id)
        factor_graph.var_nodes[p.id] = p

    # Connect dynamics factors between timestep i and i+1 and connect each dyn. factor to our parameters
    for i in range(len(t)):
        if i+1 < len(t):
            dyn_id = (i, i+1)
            factor_graph.factor_nodes[dyn_id] = DynamicsFactor(i, i+1, torch.tensor([[sigma_dynamics ** -2]]), dyn_id, factor_graph)

            for _,p in param_dict.items():
                p.connected_factors.append(dyn_id)

    # Zero mean priors on the parameters
    for i, (_,p) in enumerate(param_dict.items()):
        factor_graph.factor_nodes[i + len(t)] = ObservationFactor(i + len(t), p.id, torch.zeros((1,)), torch.tensor([[sigma_prior ** -2]]),
                                                                  factor_graph)

    # == RUN GBP (Sweep schedule) === #
    for iter in range(iters):
        # print('Iter', iter)
        print(f'Iteration {iter}, currently at {param_dict["k"].mean.item()}+-{param_dict["k"].cov.item()}')
        if iter == 0:
            # Initialise messages from observation factors to variables
            # and prior factors to parameters (if learning params)
            factor_graph.update_all_observational_factors()

            # Now update messages from variables to factors
            # This should ensure all var to dynamics factor messages have non-zero precision
            for i in factor_graph.var_nodes:
                curr = factor_graph.var_nodes[i]
                factor_graph.update_variable_belief(i)
                curr.compute_and_send_messages()

        # -- RIGHT PASS --
        for i in range(len(t)):
            curr = factor_graph.var_nodes[i]
            factor_graph.update_variable_belief(i)
            curr.compute_and_send_messages()

            if curr.right_id == -1: continue

            fac = factor_graph.factor_nodes[curr.right_id]
            factor_graph.update_factor_belief(curr.right_id)
            fac.compute_and_send_messages()

        factor_graph.update_params()

        # -- LEFT PASS --
        for i in range(len(t)-1, -1, -1):
            curr = factor_graph.var_nodes[i]
            factor_graph.update_variable_belief(i)
            curr.compute_and_send_messages()

            if curr.left_id == -1: continue

            fac = factor_graph.factor_nodes[curr.left_id]
            factor_graph.update_factor_belief(curr.left_id)
            fac.compute_and_send_messages()

        factor_graph.update_params()

        factor_graph.update_all_beliefs()


    # Plotting results
    ax.plot(signal, label='Original Signal')
    recons_signal = torch.tensor([v.mean for k, v in factor_graph.var_nodes.items() if k not in factor_graph.param_ids])

    ax.plot(recons_signal, label='GBP Result')

    ax.plot(simulate_signal(T, dt, param_dict["k"].mean.item()) + noise, label=f'Reconstructed Signal, k = {param_dict["k"].mean.item()}')
    plt.legend()
    plt.show()
