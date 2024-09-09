from fg.variables import Variable, Parameter
from fg.factors import DynamicsFactor, ObservationFactor, PriorFactor, AggregationFactor
from fg.simulation_config import simulate_hopf
from fg.graph import Graph
from fg.gaussian import Gaussian
import torch
from collections import defaultdict
import numpy as np
import copy
import sys
import pickle
import random

if __name__ == "__main__":
    sigma_obs = 1e-1
    sigma_dynamics = 5e-3
    sigma_prior = 1e1
    iters = 10
    T = 1.5
    nr = 3
    dt = 0.01
    log_every_n_iters = 1

    run_id = sys.argv[1]

    schedule = 'Hybrid'

    C = torch.empty((nr, nr)).normal_(0.7, 0.3)
    C.fill_diagonal_(0.)
    
    results_mean = defaultdict(lambda: [])
    results_cov = defaultdict(lambda: [])
    results_E = defaultdict(lambda: [])
    results_I = defaultdict(lambda: [])

    gt_config = {
        'T': T,
        'dt': dt,
        'nr': nr,
        'C': C,
        'a': torch.empty((nr,)).uniform_(0.1, 10),
        'omega': torch.empty((nr,)).uniform_(2*np.pi, 9*np.pi),
        'beta': torch.empty((nr,)).uniform_(0.1, 10),
        'obs_noise': 0.0,
    }

    config = copy.deepcopy(gt_config)

    E, I = simulate_hopf(config)
    time = torch.arange(0, len(E), 1)
    
    factor_graph = Graph(nr)

    param_list = ['a', 'omega']

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
            factor_graph.factor_nodes[f'{p_id}_prior'] = PriorFactor(
                factor_id = f'{p_id}_prior',
                var_id = p_id,
                z = torch.tensor([[1.]]).T, 
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
                    conn = torch.zeros_like(C),
                    lmbda_in = torch.tensor([[sigma_dynamics ** -2., 0.], [0., sigma_dynamics ** -2.]]),
                    factor_id = dyn_id, 
                    graph = factor_graph,
                    connected_params = [f'p({p})_r{r}' for p in param_list]
                )
    
    
    # === RUN GBP (Sweep schedule) === #
    for iter in range(iters):
        if iter % log_every_n_iters == 0:
            for r in range(nr):
                for a in param_list:
                    results_mean[r, a].append(torch.abs((gt_config[a][r] - factor_graph.var_nodes[f'p({a})_r{r}'].belief.mean.item()) / gt_config[a][r]))
                    results_cov[r, a].append(factor_graph.var_nodes[f'p({a})_r{r}'].belief.cov.item())

                    if factor_graph.var_nodes[f'p({a})_r{r}'].belief.eta.isnan().any(): 
                        print('Found nan, exiting..')
                        exit(0)
            
            # Recreate the signal with the learnt parameters and store the MAE
            for k in param_list:
                for r in range(nr):
                    t = f'p({k})_r{r}'
                    config[k][r] = factor_graph.get_var_belief(t).mean

            config['beta'] = torch.empty((nr,)).normal_(0., 0.)
            E_rec, I_rec = simulate_hopf(config)
            
            for r in range(nr):
                results_E[r].append(np.mean(np.square(E_rec[:, r] - E[:, r])))
                results_I[r].append(np.mean(np.square(I_rec[:, r] - I[:, r])))


        if iter == 0:
            factor_graph.update_all_observational_factors()

            for i in factor_graph.var_nodes:
                curr = factor_graph.var_nodes[i]
                curr.compute_and_send_messages()


        # Run the different schedule options that we have
        if schedule == 'Random':
            for i in factor_graph.var_nodes:
                if i.__class__.__name__ == 'Variable':
                    curr = factor_graph.var_nodes[i]
                    curr.compute_and_send_messages()
            
            for i in factor_graph.factor_nodes:
                curr = factor_graph.factor_nodes[i]
                curr.compute_and_send_messages() 

            factor_graph.update_params()
            for j in factor_graph.param_ids: factor_graph.factor_nodes[f'{j}_prior'].belief = factor_graph.var_nodes[j].belief
            factor_graph.update_all_observational_factors()


        elif schedule == 'Sweep':
            # Right Pass
            for t in range(len(time)-1):
                for r in range(nr):
                    curr = factor_graph.var_nodes[f'osc_t{t}_r{r}']
                    curr.compute_and_send_messages()

                    if t+1 == len(time): continue
                    
                    factor_graph.factor_nodes[(f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}')].compute_and_send_messages()
                
            factor_graph.update_params() 
            for j in factor_graph.param_ids: factor_graph.factor_nodes[f'{j}_prior'].belief = factor_graph.var_nodes[j].belief
            factor_graph.update_all_observational_factors()

            # Left Pass
            for t in range(len(time)-2, 0, -1):
                for r in range(nr):
                    curr = factor_graph.var_nodes[f'osc_t{t}_r{r}']
                    curr.compute_and_send_messages()

                    if t-1 == 0: continue

                    # Update dynamical factor
                    factor_graph.factor_nodes[(f'osc_t{t-1}_r{r}', f'osc_t{t}_r{r}')].compute_and_send_messages()
                
            factor_graph.update_params()
            for j in factor_graph.param_ids: factor_graph.factor_nodes[f'{j}_prior'].belief = factor_graph.var_nodes[j].belief
            factor_graph.update_all_observational_factors()


        elif schedule == 'Hybrid':
            alpha_left, alpha_right = 0, 0,

            # Move the right pointer along
            for alpha_right in range(15, len(time), 15):
                for _ in range(10):
                    var_nodes = [(t,r) for t in range(alpha_left, alpha_right) for r in range(nr)]
                    random.shuffle(var_nodes)
                    
                    for t, r in var_nodes:
                        curr = factor_graph.var_nodes[f'osc_t{t}_r{r}']
                        curr.compute_and_send_messages()

                    factor_nodes = [(t,r) for t in range(alpha_left, alpha_right-1) for r in range(nr)]
                    random.shuffle(factor_nodes) 

                    for t, r in factor_nodes:
                        factor_graph.factor_nodes[(f'osc_t{t}_r{r}', f'osc_t{t+1}_r{r}')].compute_and_send_messages()    

                    
                    factor_graph.update_params()

            # At the end, iterate backwards
            alpha_left, alpha_right = len(time)-1, len(time)-1
            for alpha_left in range(len(time), 0, -15):
                for _ in range(10):
                    var_nodes = [(t,r) for t in range(alpha_left, alpha_right) for r in range(nr)]

                    random.shuffle(var_nodes)
                    for t, r in var_nodes:
                        curr = factor_graph.var_nodes[f'osc_t{t}_r{r}']
                        curr.compute_and_send_messages()

                    factor_nodes = [(t,r) for t in range(alpha_left+1, alpha_right) for r in range(nr)]
                    random.shuffle(factor_nodes) 

                    for t, r in factor_nodes:
                        factor_graph.factor_nodes[(f'osc_t{t-1}_r{r}', f'osc_t{t}_r{r}')].compute_and_send_messages() 
                    
                    factor_graph.update_params()

            for j in factor_graph.param_ids: factor_graph.factor_nodes[f'{j}_prior'].belief = factor_graph.var_nodes[j].belief
            factor_graph.update_all_observational_factors()                
       
    # Combine all dictionaries into 1
    res = {
        'res_mean': dict(results_mean),
        'res_cov': dict(results_cov),
        'res_I': dict(results_E),
        'res_E': dict(results_I),
    }

    with open(f'report_results_hopf_dualhybrid_3nr_10iters_C0/abcd_{iters}iters_{schedule}_run{run_id}.pkl', 'wb') as f:
        pickle.dump(res, f)