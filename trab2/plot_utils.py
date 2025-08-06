import numpy as np
import matplotlib.pyplot as plt
from probabilities import p_hit, p_short, p_random, p_max, calc_full_probabilities

def plot_probabilities(z_exp, z_max, sigma, lbd):
    """
    Plota as distribuições de probabilidade para os parâmetros dados.
    """
    z_values = np.linspace(0, z_max + 1, 1000)
    p_hit_values = [p_hit(z, z_exp, sigma) for z in z_values]
    p_short_values = [p_short(z, z_exp, lbd) for z in z_values]
    p_random_values = [p_random(z, z_max) for z in z_values]
    p_max_values = [p_max(z, z_max) for z in z_values]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(z_values, p_hit_values, color='blue')
    axs[0, 0].set_title('Hit Probability')
    axs[0, 0].set_xlabel('z values')
    axs[0, 0].set_ylabel('Probability Density')
    axs[0, 0].grid(True)

    axs[0, 1].plot(z_values, p_short_values, color='orange')
    axs[0, 1].set_title('Short Probability')
    axs[0, 1].set_xlabel('z values')
    axs[0, 1].set_ylabel('Probability Density')
    axs[0, 1].grid(True)

    axs[1, 0].plot(z_values, p_random_values, color='green')
    axs[1, 0].set_title('Random Probability')
    axs[1, 0].set_xlabel('z values')
    axs[1, 0].set_ylabel('Probability Density')
    axs[1, 0].grid(True)

    axs[1, 1].plot(z_values, p_max_values, color='red')
    axs[1, 1].set_title('Max Probability')
    axs[1, 1].set_xlabel('z values')
    axs[1, 1].set_ylabel('Probability Density')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_combined_probabilities(z_exp, z_max, params):
    """
    Plota a combinação linear das distribuições de probabilidade.
    """
    sigma = params['sigma']
    lbd = params['lambda']

    z_values = np.linspace(0, z_max + 1, 1000)
    ph_val = [p_hit(z, z_exp, sigma) for z in z_values]
    ps_val = [p_short(z, z_exp, lbd) for z in z_values]
    pm_val = [p_max(z, z_max) for z in z_values]
    pr_val = [p_random(z, z_max) for z in z_values]

    prob_total = calc_full_probabilities(ph_val, ps_val, pm_val, pr_val, params)

    plt.figure(figsize=(10, 6))
    plt.plot(z_values, prob_total, label='Combined Probability', color='purple')
    plt.title('Combined Probability Distribution')
    plt.xlabel('z values')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_map(image):

    
    plt.imshow(image)
    plt.title('Map')
    plt.grid(True)
    plt.axis('on')


def visualize_distance_field(distance_field, walls, origin, resolution=0.05):
    """
    Shows the distance to nearest obstacle (likelihood field).
    """
    ox, oy = origin
    plt.figure(figsize=(8, 6))
    plt.imshow(distance_field, origin='lower', cmap='plasma', extent=[
        0, distance_field.shape[1] * resolution,
        0, distance_field.shape[0] * resolution
    ])
    plt.colorbar(label='Distance (m)')

    for (x1, y1), (x2, y2) in walls:
        plt.plot([x1 - ox, x2 - ox], [y1 - oy, y2 - oy], 'k-', linewidth=2)

    plt.title("Distance Field + Wall Segments")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(False)
    plt.axis("equal")
    #plt.show()

def visualize_pose_with_measurements(pose, scan, walls, origin, distance_field, beam_angles, resolution=0.05, color='red'):
    """
    Draws the scan rays from a pose over the occupancy map and distance field.
    """
    ox, oy = origin
    x, y, theta = pose

    plt.figure(figsize=(8, 8))
    h, w = int(10 / resolution), int(10 / resolution)  # match your occupancy grid bounds
    extent = [0, w * resolution, 0, h * resolution]

    plt.imshow(distance_field, origin='lower', cmap='plasma', extent=extent)
    plt.colorbar(label='Distance to nearest obstacle (m)')

    for d, a in zip(scan, beam_angles):
        if d == 0.0 or np.isnan(d): continue
        x_end = x + d * np.cos(theta + a)
        y_end = y + d * np.sin(theta + a)
        plt.plot([x, x_end], [y, y_end], color=color, alpha=0.6)

    # Draw walls
    for (x1, y1), (x2, y2) in walls:
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    plt.plot(x, y, 'bo')  # robot
    plt.title("Scan Rays with Walls and Likelihood Field")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def viz_likelihood(norm_likelihood,origin,resolution,grid,walls):

     # Plot the likelihood field
    ox, oy = origin
    plt.figure(figsize=(8, 6))
    plt.imshow(norm_likelihood, cmap='viridis', origin='lower', extent=[
        origin[0], origin[0] + grid.shape[1] * resolution,
        origin[1], origin[1] + grid.shape[0] * resolution
    ])
    for (x1, y1), (x2, y2) in walls:
        plt.plot([x1 - ox, x2 - ox], [y1 - oy, y2 - oy], 'k-', linewidth=2)
    plt.colorbar(label='Normalized Likelihood')
    plt.title("Range Finder Likelihood Field")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    z_exp = 20
    z_max = 40
    sigma = 1
    lbd = 0.05
    plot_probabilities(z_exp, z_max, sigma, lbd)
    print("Plots generated for the probability distributions.")