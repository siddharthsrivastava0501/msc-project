import re
import matplotlib.pyplot as plt
import numpy as np

def parse_debug_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Extract ground truth values
    match = re.search(r'a = tensor\(\[(.*?)\]\), b = tensor\(\[(.*?)\]\), c = tensor\(\[(.*?)\]\), d = tensor\(\[(.*?)\]\)', content)
    ground_truth = list(map(float, match.groups()))

    # Initialize data structures
    iterations = []
    mu_data = {'a': [], 'b': [], 'c': [], 'd': [], 'P' : [], 'Q': []}
    cov_data = {'a': [], 'b': [], 'c': [], 'd': [], 'P' : [], 'Q': []}

    # Parse iterations
    for iteration_match in re.finditer(r'Iteration (\d+)(.*?)(?=Iteration|\Z)', content, re.DOTALL):
        iteration = int(iteration_match.group(1))
        iterations.append(iteration)
        iteration_content = iteration_match.group(2)

        # Parse parameter values
        for param in ['a', 'b', 'c', 'd', 'P', 'Q']:
            param_match = re.search(rf'Parameter p\({param}\)_r0 \[n = 1, mu=tensor\(\[\[(.*?)\]\]\), cov=tensor\(\[\[(.*?)\]\]\)\]', iteration_content)
            if param_match:
                mu_data[param].append(float(param_match.group(1)))
                cov_data[param].append(float(param_match.group(2)))
            else:
                mu_data[param].append(None)
                cov_data[param].append(None)

    return ground_truth, iterations, mu_data, cov_data

# Parse the debug file
ground_truth, iterations, mu_data, cov_data = parse_debug_file('debug.txt')

# Create the plot
fig, axes = plt.subplots(2, 3, figsize=(15, 15))
fig.suptitle('Parameter Estimation over Iterations', fontsize=16)

parameters = ['a', 'b', 'c', 'd', 'P', 'Q']

for idx, param in enumerate(parameters):
    ax1 = axes[idx // 2, idx % 2]
    ax2 = ax1.twinx()
    
    # Plot mu
    ax1.plot(iterations, mu_data[param], 'b-', label='mu')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('mu', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot cov
    ax2.plot(iterations, cov_data[param], 'r-', label='cov')
    ax2.set_ylabel('cov', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')  # Use log scale for covariance
    
    # Plot ground truth
    ax1.axhline(y=ground_truth[idx], color='g', linestyle='--', label='Ground Truth')
    
    ax1.set_title(f'Parameter {param}')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.show()