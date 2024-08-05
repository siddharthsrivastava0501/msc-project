from fg.variables import Variable, Parameter
from fg.factors import DynamicsFactor, ObservationFactor, PriorFactor, AggregationFactor
from fg.simulation_config import simulate_wc
from fg.graph import Graph
from fg.gaussian import Gaussian
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sigma_obs = 1e-3
    sigma_dynamics = 1e-3
    sigma_prior = 5e0
    iters = 100
    T = 12
    nr = 1
    dt = 0.01

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
        'P': torch.empty((nr,)).normal_(1., 0.1),
        'Q': torch.empty((nr,)).normal_(1., 0.1),
    }

    E, I = simulate_wc(config)
    time = torch.arange(0, T + dt, dt)
    plt.plot(E, label='GT E')
    plt.plot(I, label='GT I')
    plt.legend()
    plt.show()
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    factor_graph = Graph()

    param_list = ['a', 'b', 'c', 'd', 'P', 'Q']
    
    # Create the E-I oscillators and add priors to those oscillators
    for t in range(len(time)):
        for r in range(nr):
            factor_graph.var_nodes[f'osc_t{t}_r{r}'] = Variable(
                id       = f'osc_t{t}_r{r}',
                belief   = Gaussian(torch.tensor([[0.1, 0.1]]).T, torch.tensor([[0.2, 0.], [0., 0.2]])),
                graph    = factor_graph, 
                num_vars = 2,
                connected_factors = [f'agg_t{t}_r{r_id}' for r_id in range(nr) if r_id != r] if t+1 < len(time) else []
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
                z = torch.tensor([[3.]]).T, 
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
                    Sigma_id = f'in_t{t}_r{r}', 
                    lmbda_in = torch.tensor([[sigma_dynamics ** -2]]),
                    factor_id = dyn_id, 
                    graph = factor_graph,
                    connected_params = [f'p({p})_r{r}' for p in param_list]
                )

                # For the two nodes that we have just connected, add this dyn. factor
                # as an adjacent connected factor
                factor_graph.var_nodes[f'osc_t{t}_r{r}'].connected_factors.append(dyn_id)
                factor_graph.var_nodes[f'osc_t{t+1}_r{r}'].connected_factors.append(dyn_id)


    # Create the input variables and the aggregation factor for inter-region connections
    for t in range(len(time)):
        for r in range(nr):
            if t+1 < len(time):
                # Create input variables that connect to each dynamic factor
                factor_graph.var_nodes[f'in_t{t}_r{r}'] = Variable(
                    id = f'in_t{t}_r{r}',
                    belief = Gaussian(torch.tensor([[0.]]), torch.tensor([[0.2]])), 
                    graph = factor_graph,

                    # Connect this variable to its aggregation factor and its corresponding 
                    connected_factors = [f'agg_t{t}_r{r}', (f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}')]
                )

                # Add priors to the input variables for stability
                factor_graph.factor_nodes[f'p_in_t{t}_r{r}'] = PriorFactor(
                    factor_id = f'p_in_t{t}_r{r}',
                    var_id = f'in_t{t}_r{r}',
                    z = torch.tensor([[0., 0.]]).T, 
                    lmbda_in =  torch.tensor([[0.1 ** -2, 0.], [0., 0.1 ** -2]]),
                    graph = factor_graph
                )

                # Add nr-ary aggregation factor between our nr-1 regions and \Sigma var.
                factor_graph.factor_nodes[f'agg_t{t}_r{r}'] = AggregationFactor(
                    factor_id = f'agg_t{t}_r{r}',
                    region_id = r,
                    input_id = f'in_t{t}_r{r}',
                    C = C,
                    lmbda_in = torch.tensor([[1e-3 ** -2]]),
                    graph = factor_graph,
                    connected_regions = [f'osc_t{t}_r{i}' for i in range(nr) if i != r]
                )

    def process_oscillator(t, r, factor_graph):
        curr = factor_graph.var_nodes[f'osc_t{t}_r{r}']
        curr.compute_and_send_messages()

    def process_aggregation_and_input(t, r, factor_graph):
        factor_graph.factor_nodes[f'agg_t{t}_r{r}'].compute_and_send_messages()  
        factor_graph.var_nodes[f'in_t{t}_r{r}'].compute_and_send_messages()

    def process_dynamics(t, r, factor_graph):
        factor_graph.factor_nodes[(f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}')].compute_and_send_messages()
    
    def parallel_loop(func, nr, t, factor_graph):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(func, t, r, factor_graph) for r in range(nr)]
            for future in as_completed(futures):
                future.result()  # This will raise an exception if one occurred during the execution
    
    for iter in range(iters):
        print(f'Iteration {iter}')
        for r in range(nr):
            for a in param_list:
                print(factor_graph.var_nodes[f'p({a})_r{r}'])

            if factor_graph.var_nodes[f'p({p})_r{r}'].belief.eta.isnan().any(): 
                exit(0)
        
        if iter == 0:
            factor_graph.update_all_observational_factors()
            for i in factor_graph.var_nodes:
                curr = factor_graph.var_nodes[i]
                curr.compute_and_send_messages()
            
            factor_graph.prune() 

        # Right Pass
        for t in range(len(time)-1):
            # Parallelize oscillator processing
            parallel_loop(process_oscillator, nr, t, factor_graph)

            # Parallelize aggregation and input processing
            parallel_loop(process_aggregation_and_input, nr, t, factor_graph)

            if t+1 < len(time):
                # Parallelize dynamics processing
                parallel_loop(process_dynamics, nr, t, factor_graph)
        
        factor_graph.update_params() 

        # Left Pass
        for t in range(len(time)-2, 0, -1):
            # Parallelize oscillator processing
            parallel_loop(process_oscillator, nr, t, factor_graph)

            # Parallelize aggregation and input processing
            parallel_loop(process_aggregation_and_input, nr, t, factor_graph)

            if t-1 > 0:
                # Parallelize dynamics processing
                parallel_loop(process_dynamics, nr, t-1, factor_graph)
        
        factor_graph.update_params()

    E_rec, I_rec = simulate_wc(config)

    plt.plot(E, label='GT E')
    plt.plot(I, label='GT I')
    plt.plot(E_rec.detach().numpy(), label='E_rec')
    plt.plot(I_rec.detach().numpy(), label='I_rec')
    plt.legend()
    plt.show()
