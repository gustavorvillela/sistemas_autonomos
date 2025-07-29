import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

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

    def plot_velocity_particles(x_prev, x_next_est, particles, r=0.1, title="Velocity Motion Model - Particles"):
        plt.figure(figsize=(8, 6))
        plt.title(title)
        plt.xlabel("Position x")
        plt.ylabel("Position y")
        plt.grid(True)

        # Partículas
        plt.scatter(particles[:, 0], particles[:, 1], s=1, color='yellow', alpha=0.5, label="Particles")

        # Robô anterior (círculo + orientação)
        circle_prev = Circle((x_prev[0], x_prev[1]), r, color='green', ec='black', alpha=0.7, label='Previous Robot')
        plt.gca().add_patch(circle_prev)
        plt.arrow(x_prev[0], x_prev[1], r * np.cos(x_prev[2]), r * np.sin(x_prev[2]),
                  head_width=0.03, fc='black', ec='black')

        # Robô estimado (círculo + orientação)
        circle_est = Circle((x_next_est[0], x_next_est[1]), r, color='cyan', ec='black', alpha=0.7,
                            label='Expected Robot')
        plt.gca().add_patch(circle_est)
        plt.arrow(x_next_est[0], x_next_est[1], r * np.cos(x_next_est[2]), r * np.sin(x_next_est[2]),
                  head_width=0.03, fc='magenta', ec='magenta')

        # Linha entre poses
        plt.plot([x_prev[0], x_next_est[0]], [x_prev[1], x_next_est[1]], 'r--', linewidth=1, label="Motion")

        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_velocity_particles(x_prev, x_next_est, particles, r=0.1, title="Velocity Motion Model - Particles"):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.grid(True)

    # Partículas
    plt.scatter(particles[:, 0], particles[:, 1], s=1, color='yellow', alpha=0.5, label="Particles")

    # Robô anterior (círculo + orientação)
    circle_prev = Circle((x_prev[0], x_prev[1]), r, color='green', ec='black', alpha=0.7, label='Previous Robot')
    plt.gca().add_patch(circle_prev)
    plt.arrow(x_prev[0], x_prev[1], r*np.cos(x_prev[2]), r*np.sin(x_prev[2]),
              head_width=0.03, fc='black', ec='black')

    # Robô estimado (círculo + orientação)
    circle_est = Circle((x_next_est[0], x_next_est[1]), r, color='cyan', ec='black', alpha=0.7, label='Expected Robot')
    plt.gca().add_patch(circle_est)
    plt.arrow(x_next_est[0], x_next_est[1], r*np.cos(x_next_est[2]), r*np.sin(x_next_est[2]),
              head_width=0.03, fc='magenta', ec='magenta')

    # Linha entre poses
    plt.plot([x_prev[0], x_next_est[0]], [x_prev[1], x_next_est[1]], 'r--', linewidth=1, label="Motion")

    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()