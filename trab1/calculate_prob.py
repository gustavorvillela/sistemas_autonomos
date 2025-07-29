import numpy as np

def prob_normal_distribution(a,b2):
    """
    Calculate the probability of a sample from a normal distribution with Mahalanobis distance a
    and variance b2.
    
    Parameters:
    a (float): Mahalanobis distance.
    b2 (float): Variance of the normal distribution.
    
    Returns:
    prob (float): Probability of the sample.
    """
    return (1/(2*np.pi*b2)) * np.exp(-0.5 * (a**2 / b2))

def prob_triangular_distribution(a,b2):
    """
    Calculate the probability of a sample from a triangular distribution with Mahalanobis distance a
    and variance b2.
    
    Parameters:
    a (float): Mahalanobis distance.
    b2 (float): Variance of the triangular distribution.
    
    Returns:
    prob (float): Probability of the sample.
    """
    b = np.sqrt(b2)

    return np.max([0, 1/(np.sqrt(6) * b) * (1 - abs(a)/(np.sqrt(6) * b))])