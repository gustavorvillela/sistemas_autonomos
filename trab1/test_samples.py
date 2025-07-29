import numpy as np
from generate_samples import sample_normal_distribution, generate_triangular_distribution

import matplotlib.pyplot as plt

def generate_normal_samples(N,b2=1):
    """
    Generate N samples from a normal distribution with mean 0 and variance 1.
    
    Parameters:
    N (int): Number of samples to generate.
    
    Returns:
    samples (list): List of samples from the normal distribution.
    """
    return [sample_normal_distribution(b2) for _ in range(N)]

def generate_triangular_samples(N,b2=1):
    """
    Generate N samples from a triangular distribution with mean 0 and variance 1.
    
    Parameters:
    N (int): Number of samples to generate.
    
    Returns:
    samples (list): List of samples from the triangular distribution.
    """
    return [generate_triangular_distribution(b2) for _ in range(N)]

def plot_samples(samples, title, figure_num):
    plt.figure(figure_num)
    plt.hist(samples, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

if __name__ == "__main__":
    N = 100000  # Number of particles

    # Generate samples
    normal_samples = generate_normal_samples(N,2)
    triangular_samples = generate_triangular_samples(N,2)

    # Plot samples
    plot_samples(normal_samples, "Normal Distribution", 1)
    plot_samples(triangular_samples, "Triangular Distribution", 2)

    # Show plots
    plt.show()