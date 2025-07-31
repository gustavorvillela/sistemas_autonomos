'''''import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import  Circle
from generate_samples import sample_normal_distribution

def sample_motion_model_odometry(x_prev, odo_prev, odo_curr, alphas, num_samples=1000):
    """Gera partículas de acordo com o modelo de movimento por odometria"""
    particles = []

    for _ in range(num_samples):
        x, y, theta = x_prev
        xb, yb, thetab = odo_prev
        xbp, ybp, thetabp = odo_curr

        delta_rot1 = np.arctan2(ybp - yb, xbp - xb) - thetab
        delta_trans = np.sqrt((xbp - xb)**2 + (ybp - yb)**2)
        delta_rot2 = thetabp - thetab - delta_rot1

        drot1_hat = delta_rot1 - sample_normal_distribution(alphas[0] * delta_rot1**2 + alphas[1] * delta_trans**2)
        dtrans_hat = delta_trans - sample_normal_distribution(alphas[2] * delta_trans**2 + alphas[3] * (delta_rot1**2 + delta_rot2**2))
        drot2_hat = delta_rot2 - sample_normal_distribution(alphas[4] * delta_rot2**2 + alphas[5] * delta_trans**2)

        x_sample = x + dtrans_hat * np.cos(theta + drot1_hat)
        y_sample = y + dtrans_hat * np.sin(theta + drot1_hat)
        theta_sample = theta + drot1_hat + drot2_hat

        particles.append((x_sample, y_sample, theta_sample))

    return np.array(particles)


def plot_odometry_particles(x_prev, x_next_est, particles, r=0.1, title="Odometry Motion Model - Particles"):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.grid(True)

    # Partículas em amarelo
    plt.scatter(particles[:, 0], particles[:, 1], s=1, color='yellow', alpha=0.5, label="Particles")

    # Robô anterior (círculo com orientação)
    circle_prev = Circle((x_prev[0], x_prev[1]), r, color='green', ec='black', alpha=0.7, label='Previous Robot')
    plt.gca().add_patch(circle_prev)
    plt.arrow(x_prev[0], x_prev[1], r * np.cos(x_prev[2]), r * np.sin(x_prev[2]),
              head_width=0.03, fc='black', ec='black')

    # Robô estimado (círculo com orientação)
    circle_est = Circle((x_next_est[0], x_next_est[1]), r, color='cyan', ec='black', alpha=0.7, label='Expected Robot')
    plt.gca().add_patch(circle_est)
    plt.arrow(x_next_est[0], x_next_est[1], r * np.cos(x_next_est[2]), r * np.sin(x_next_est[2]),
              head_width=0.03, fc='magenta', ec='magenta')

    # Linha entre poses
    plt.plot([x_prev[0], x_next_est[0]], [x_prev[1], x_next_est[1]], 'r--', linewidth=1, label="Motion")

    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    x_prev = (0.0, 0.0, 0.0)
    x_real = (1.0, 0.5, np.pi/8)

    # Supomos que a odometria media algo próximo ao real com ruído
    odo_prev = (0.0, 0.0, 0.0)
    odo_curr = (1.05, 0.55, np.pi/7)

    alphas = [0.1]*6
    particles = sample_motion_model_odometry(x_prev, odo_prev, odo_curr, alphas, num_samples=2000)

    plot_odometry_particles(x_prev, x_real, particles, r=0.1)
'''''''''
import numpy as np
from motion_model import odometry_prob_grid, generate_motion_model_particles
from plot_utils import plot_odometry_particles,generate_grid,plot_prob_odom, plot_prob_velocity
# Make sure generate_samples.py exists in the same directory or is accessible in your Python path
from generate_samples import sample_normal_distribution

if __name__ == "__main__":
    x_prev = (0.0, 0.0, 0.0)
    x_real = (1.0, 0.5, np.pi/8)

    # Supomos que a odometria media algo próximo ao real com ruído
    odo_prev = (0.0, 0.0, 0.0)
    odo_curr = (1.05, 0.55, np.pi/7)

    #alphas = [0.1]*6
    alphas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Parâmetros de ruído do modelo de movimento
    particles = generate_motion_model_particles(x_prev, odo_prev, odo_curr, alphas, num_samples=10000)

    plot_odometry_particles(x_prev, x_real, particles, r=0.1)

    # Plot probabilities

    grid = generate_grid((-2, 2), (-2, 2), num_points=100)
    probabilities = odometry_prob_grid(grid, x_prev,x_real[2], odo_prev, odo_curr, alphas)

    plot_prob_velocity(grid, probabilities, x_prev, x_real, r=0.1, title="Odometry Probability Distribution")