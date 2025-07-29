import numpy as np
import random

def sample_normal_distribution(b2):

    """
    Generate samples from a normal distribution with mean 0 and variance b2.
    
    Parameters:
    b2 (float): Standard deviation of the normal distribution.
    
    Returns:
    s (float): A sample from the normal distribution. 
    """

    b = np.sqrt(b2)

    rands = np.array([random.uniform(-b,b) for _ in range(12)])
    s = 0.5 * np.sum(rands)  # Central limit theorem: sum of uniform distributions approximates normal distribution
    
    return s

def generate_triangular_distribution(b2):

    """
    Generate samples from a triangular distribution with mean 0 and variance b2.
    
    Parameters:
    b2 (float): Variance of the triangular distribution.
    
    Returns:
    s (float): A sample from the triangular distribution. 
    """

    b = np.sqrt(b2)
    
    rands = random.uniform(-b, b) + random.uniform(-b, b)
    s = np.sqrt(6) / 2 * rands  # Scaling to match the variance of the triangular distribution
    
    return s