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
    plt.scatter(particles[:, 0], particles[:, 1], s=1, color='blue', alpha=0.5, label="Particles")

    

    add_robot_mov(x_prev, x_next_est, r)

    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    #plt.show()


def add_robot_mov(x_prev, x_next_est,r=0.1):

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


def generate_grid(x_lim, y_lim, num_points=100):
    """
    Gera uma grade de pontos dentro dos limites especificados.
    
    Parameters:
    x_lim (tuple): Limites do eixo x (min, max).
    y_lim (tuple): Limites do eixo y (min, max).
    num_points (int): Número de pontos na grade.
    
    Returns:
    np.ndarray: Array de pontos na grade.
    """
    x = np.linspace(x_lim[0], x_lim[1], num_points)
    y = np.linspace(y_lim[0], y_lim[1], num_points)

    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

def plot_prob_odom(grid_points, probabilities, x_prev, x_next_est, r=0.1, title="Distribuição de Probabilidade"):
    """
    Plota a distribuição de probabilidade em uma grade de pontos.
    
    Parameters:
    grid_points (np.ndarray): Array de pontos na grade.
    title (str): Título do gráfico.
    """
    plt.figure(figsize=(6, 6))
    plt.tricontourf(grid_points[:, 0], grid_points[:, 1], probabilities, levels=100, cmap='viridis')
    plt.colorbar(label='Probability Density')
    add_robot_mov(x_prev, x_next_est, r)
    plt.title(title)
    plt.xlim(grid_points[:, 0].min(), grid_points[:, 0].max())
    plt.ylim(grid_points[:, 1].min(), grid_points[:, 1].max())
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')

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