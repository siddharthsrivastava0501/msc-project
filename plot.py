import re
import matplotlib.pyplot as plt

# Function to extract data from the text
def extract_data(text):
    parameters = ['k1', 'k2', 'k3', 'k4', 'P', 'Q']
    data = {param: {'mu': [], 'cov': []} for param in parameters}

    for line in text.split('\n'):
        for param in parameters:
            if line.startswith(f'{param} Parameter'):
                mu = float(re.search(r'mu=tensor\(\[\[(.*?)\]\]', line).group(1))
                cov = float(re.search(r'cov=tensor\(\[\[(.*?)\]\]', line).group(1))
                data[param]['mu'].append(mu)
                data[param]['cov'].append(cov)

    return data

# Extract ground truth parameters
def extract_ground_truth(text):
    params = re.findall(r'(\w+) = ([\d.]+)', text)
    return {k: float(v) for k, v in params}

# Extract data
with open('debug.txt', 'r') as file:
    text = file.read()

data = extract_data(text)
ground_truth = extract_ground_truth(text.split('\n')[0])

# Create 6 separate plots
fig, axs = plt.subplots(3, 2, figsize=(20, 15))
fig.suptitle('Parameter Evolution over Iterations', fontsize=16)

for i, (param, values) in enumerate(data.items()):
    row = i // 2
    col = i % 2

    ax1 = axs[row, col]
    ax2 = ax1.twinx()

    iterations = range(len(values['mu']))

    # Plot mu
    line1, = ax1.plot(iterations, values['mu'], 'b-', label='mu')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('mu', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot cov
    line2, = ax2.plot(iterations, values['cov'], 'r-', label='cov')
    ax2.set_ylabel('cov', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add ground truth line
    if param in ground_truth:
        line3 = ax1.axhline(y=ground_truth[param], color='g', linestyle='--', label='Ground Truth')

    ax1.set_title(f'{param} Parameter')

    # Add legend
    lines = [line1, line2, line3] if param in ground_truth else [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines])

plt.tight_layout()
plt.show()
