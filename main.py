from fg.variables import Variable, Parameter
from fg.factors import DynamicsFactor, ObservationFactor, PriorFactor
from fg.simulation_config import simulate_wc
from fg.graph import Graph
from fg.gaussian import Gaussian
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import random

if __name__ == "__main__":
    sigma_obs = 1e-2
    sigma_dynamics = 1e-3
    sigma_prior = 1e1
    iters = 300
    T = 8
    nr = 10
    dt = 0.05

    C = torch.empty((nr, nr)).normal_(0.2, 0.1)
    C.fill_diagonal_(0.)

    config = {
        'T': T,
        'dt': dt,
        'nr': nr,
        'C': C,
        'a': torch.empty((nr,)).normal_(3., 1.),
        'b': torch.empty((nr,)).normal_(5., 1.),
        'c': torch.empty((nr,)).normal_(4., 1.),
        'd': torch.empty((nr,)).normal_(3., 1.),
        'P': torch.empty((nr,)).normal_(1., 0.),
        'Q': torch.empty((nr,)).normal_(1., 0.),
    }

    E, I = simulate_wc(config)
    time = torch.arange(0, len(E), 1)
    plt.plot(E)
    plt.plot(I)
    plt.show()

    factor_graph = Graph(nr)

    param_list = ['a', 'b', 'c', 'd']

    # -- Construct FG -- #
    # Add our variable and observation factors at each time step
    for t in range(len(time)):
        for r in range(nr):
            factor_graph.var_nodes[f'osc_t{t}_r{r}'] = Variable(
                id       = f'osc_t{t}_r{r}',
                belief   = Gaussian(torch.tensor([[0.1, 0.1]]).T, torch.tensor([[0.2, 0.], [0., 0.2]])),
                graph    = factor_graph, 
                num_vars = 2,
                connected_factors = [(f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}') if t+1 < len(time) else -1] +  [(f'osc_t{t-1}_r{r}', f'osc_t{t}_r{r}') if t > 0 else -1]
            )
            
            factor_graph.factor_nodes[f'obs_t{t}_r{r}'] = ObservationFactor(
                factor_id = f'obs_t{t}_r{r}', 
                var_id    = f'osc_t{t}_r{r}',
                z         = torch.tensor([[E[t, r], I[t, r]]]).T.float(),
                lmbda_in  = torch.tensor([[sigma_obs ** -2, 0.], [0., sigma_obs ** -2]]),
                graph     = factor_graph
            )
        
    # Add parameters to each region
    for p in param_list:
        for r in range(nr):
            p_id = f'p({p})_r{r}'

            factor_graph.param_ids.append(p_id)      
            factor_graph.var_nodes[p_id] = Parameter(
                id     = p_id, 
                belief = Gaussian(torch.tensor([[0.]]), torch.tensor([[sigma_prior ** 2.]])),
                graph  = factor_graph,
                connected_factors = [(f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}') for t in range(len(time)-1)]
            ) 

            # Add priors to those parameters
            factor_graph.factor_nodes[f'p_prior_p{p}_r{r}'] = PriorFactor(
                factor_id = f'p_prior_p{p}_r{r}',
                var_id = p_id,
                z = torch.tensor([[0.]]).T, 
                lmbda_in = torch.diag(torch.tensor([sigma_prior ** -2])),
                graph = factor_graph
            )

    # Add the dynamics factors between timesteps in every region
    for r in range(nr):
        for t in range(len(time)):
            if t+1 < len(time):
                dyn_id = (f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}')
                factor_graph.factor_nodes[dyn_id] = DynamicsFactor(
                    Vt_id  = f'osc_t{t}_r{r}',
                    Vtp_id = f'osc_t{t+1}_r{r}',
                    region_id = r,
                    conn = C,
                    lmbda_in = torch.tensor([[sigma_dynamics ** -2]]),
                    factor_id = dyn_id, 
                    graph = factor_graph,
                    connected_params = [f'p({p})_r{r}' for p in param_list]
                )


    # === RUN GBP (Sweep schedule) === #
    for iter in range(iters):
        print(f'Iteration {iter} {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}')
        for a in param_list:
            for r in range(nr):
                print(factor_graph.var_nodes[f'p({a})_r{r}'])

                if factor_graph.var_nodes[f'p({a})_r{r}'].belief.eta.isnan().any(): 
                    exit(0)

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

        # Step 1: Update all variable nodes in random order
        var_nodes = [(t, r) for t in range(len(time)) for r in range(nr)]
        random.shuffle(var_nodes)
        
        for t, r in var_nodes:
            curr = factor_graph.var_nodes[f'osc_t{t}_r{r}']
            curr.compute_and_send_messages()
        
        # Step 2: Update all factor nodes
        factor_nodes = [(t, r) for t in range(len(time)-1) for r in range(nr)]
        random.shuffle(factor_nodes)
        
        for t, r in factor_nodes:
            factor_graph.factor_nodes[(f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}')].compute_and_send_messages()
        
        # Update parameters after each complete iteration
        factor_graph.update_params()


    # update config and plot both
    for k in param_list:
        for r in range(nr):
            t = f'p({k})_r{r}'
            config[k][r] = factor_graph.get_var_belief(t).mean

    config['dyn_noise'] = False

    E_rec, I_rec = simulate_wc(config)
    plt.plot(E, label='GT E')
    # plt.plot(I, label='GT I')
    plt.plot(E_rec, label='E rec')
    # plt.plot(I_rec, label='I rec')
    plt.legend()
    plt.show()