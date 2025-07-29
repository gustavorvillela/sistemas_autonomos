import numpy as np
import matplotlib.pyplot as plt

from generate_samples import (
    sample_normal_distribution,
    generate_triangular_distribution
)

from calculate_prob import (    
    prob_normal_distribution,
    prob_triangular_distribution
)

def generate_normal_samples(N, b2=1):
    return [sample_normal_distribution(b2) for _ in range(N)]

def generate_triangular_samples(N, b2=1):
    return [generate_triangular_distribution(b2) for _ in range(N)]

def plot_probabilities(samples, prob_func, title, figure_num, b2):
    samples = np.array(samples)
    probs = np.array([prob_func(x, b2) for x in samples])

    plt.figure(figure_num)
    plt.scatter(samples, probs, s=1, alpha=0.5, color='blue')
    plt.title(title)
    plt.xlabel('Sample Value')
    plt.ylabel('Probability Density')
    plt.grid(True)

if __name__ == "__main__":
    N = 100000
    b2 = 2

    # Generate samples
    normal_samples = generate_normal_samples(N, b2)
    triangular_samples = generate_triangular_samples(N, b2)

    # Plot using probability functions
    plot_probabilities(normal_samples, prob_normal_distribution, "Normal Distribution (Probability)", 1, b2)
    plot_probabilities(triangular_samples, prob_triangular_distribution, "Triangular Distribution (Probability)", 2, b2)

    plt.show()
