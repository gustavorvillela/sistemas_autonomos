import numpy as np
from matplotlib.patches import Ellipse
from scipy.linalg import eigh
import matplotlib.pyplot as plt



def plot_mahalanobis_distance(mu, S, N, n=3):
    # Generate random points from a multivariate normal distribution
    points = np.random.multivariate_normal(mu, S, N)
    
    # Compute the Mahalanobis distance for each point
    S_inv = np.linalg.inv(S)
    distances = [np.sqrt((p - mu).T @ S_inv @ (p - mu)) for p in points]
    
    # Filter points within the nth Mahalanobis norm
    within_n = points[np.array(distances) <= n]
    
    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = eigh(S)
    
    # Sort eigenvalues and eigenvectors (descending order)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    width_base = 2 * np.sqrt(eigenvalues[0])
    height_base = 2 * np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.3, s=5, label='All Points')
    plt.scatter(within_n[:, 0], within_n[:, 1], color='blue', alpha=0.6, s=5, label=f'Points within {n} Mahalanobis norm')
    
    edge_colors = ['red', 'orange', 'blue', 'purple', 'cyan']
    
    for i in range(1, n + 1):
        edge_color = edge_colors[(i - 1) % len(edge_colors)]
        width = i * width_base
        height = i * height_base
        ellipse = Ellipse(xy=mu, width=width, height=height,
                          angle=angle, edgecolor=edge_color,
                          facecolor='none', linestyle='--', label=f'{i} $\sigma$')
        plt.gca().add_patch(ellipse)

    # Draw principal axes (major and minor)
    for length, vec in zip([width_base / 2, height_base / 2], eigenvectors.T):
        start = mu - length * vec * n
        end = mu + length * vec * n
        plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linestyle=':', linewidth=1.5)
    
    plt.title('Mahalanobis Distance Plot with Principal Axes')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    mu = np.array([0, 0])
    S = np.array([[1, 0.5], [0.5, 1]])
    N = 2000
    n = 3
    plot_mahalanobis_distance(mu, S, N, n)
