from fg.variables import Variable, Parameter
from fg.factors import DynamicsFactor, ObservationFactor
from fg.simulation_config import simulate_signal
from fg.graph import Graph
from fg.gaussian import Gaussian
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sigma_obs = 1e-2
    sigma_dynamics = 1e-3
    GT_k = 1.2
    T, dt = 15, 0.01
    iters = 20

    signal = simulate_signal(T, dt, GT_k)
    # signal = [1., 2., 3., 4.]
    t = torch.arange(0, len(signal), 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    factor_graph = Graph()

    param_dict = {
        # 'k': Parameter(len(t), Gaussian(torch.tensor([[1.0]]), torch.tensor([[2.]])), factor_graph, [])
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


    # == RUN GBP (Sweep schedule) === #
    for iter in range(iters):
        print(f'Iteration {iter}')
        if iter == 0:
            factor_graph.send_initial_parameter_messages()

        factor_graph.update_all_observational_factors()

        # -- RIGHT PASS --
        for i in range(len(t)-1):
            curr = factor_graph.var_nodes[i]

            curr.compute_and_send_messages()

            fac = factor_graph.factor_nodes[curr.right_id]

            fac.compute_messages_except_key()

        factor_graph.update_params()

        # -- LEFT PASS --
        for i in range(len(t)-1, 0, -1):
            curr = factor_graph.var_nodes[i]

            curr.compute_and_send_messages()

            fac = factor_graph.factor_nodes[curr.left_id]

            fac.compute_messages_except_key()

        factor_graph.update_params()


    # Plotting results
    ax.plot(signal, label='Original Signal')
    recons_signal = torch.tensor([v.mean for k, v in factor_graph.var_nodes.items() if k not in factor_graph.param_ids])

    print(signal.shape, recons_signal.shape)

    ax.plot(recons_signal, label='GBP Result')
    plt.legend()
    plt.show()
