import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


# ============================
# Função auxiliar reutilizável
# ============================
def add_robot_mov(x_prev, x_next_est, r=0.1):
    """Desenha robô anterior e estimado como círculos com orientação."""

    # Robô anterior
    circle_prev = Circle((x_prev[0], x_prev[1]), r, color='green', ec='black', alpha=0.7, label='Previous Robot')
    plt.gca().add_patch(circle_prev)
    plt.arrow(x_prev[0], x_prev[1], r * np.cos(x_prev[2]), r * np.sin(x_prev[2]),
              head_width=0.03, fc='black', ec='black')

    # Robô estimado
    circle_est = Circle((x_next_est[0], x_next_est[1]), r, color='cyan', ec='black', alpha=0.7, label='Expected Robot')
    plt.gca().add_patch(circle_est)
    plt.arrow(x_next_est[0], x_next_est[1], r * np.cos(x_next_est[2]), r * np.sin(x_next_est[2]),
              head_width=0.03, fc='magenta', ec='magenta')

    # Linha entre poses
    plt.plot([x_prev[0], x_next_est[0]], [x_prev[1], x_next_est[1]], 'r--', linewidth=1, label="Motion")


# ============================
# Plotagem de partículas (odometria)
# ============================
def plot_odometry_particles(x_prev, x_next_est, particles, r=0.1, title="Odometry Motion Model - Particles"):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.grid(True)

    plt.scatter(particles[:, 0], particles[:, 1], s=1, color='blue', alpha=0.5, label="Particles")
    add_robot_mov(x_prev, x_next_est, r)

    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================
# Plotagem de partículas (velocidade)
# ============================
def plot_velocity_particles(x_prev, x_next_est, particles, r=0.1, title="Velocity Motion Model - Particles"):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.grid(True)

    plt.scatter(particles[:, 0], particles[:, 1], s=1, color='blue', alpha=0.5, label="Particles")
    add_robot_mov(x_prev, x_next_est, r)

    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    #plt.show()


# ============================
# Plotagem de mapa de probabilidade (odometria)
# ============================
def plot_prob_odom(grid_points, probabilities, x_prev, x_next_est, r=0.1, title="Odometry Model - Probability Map"):
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
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================
# Plotagem de mapa de probabilidade (velocidade)
# ============================
def plot_prob_velocity(grid_points, probabilities, x_prev, x_next_est, r=0.1, title="Velocity Model - Probability Map"):
    plt.figure(figsize=(6, 6))
    plt.tricontourf(grid_points[:, 0], grid_points[:, 1], probabilities, levels=100, cmap='plasma')
    plt.colorbar(label='Probability Density')

    add_robot_mov(x_prev, x_next_est, r)

    plt.title(title)
    plt.xlim(grid_points[:, 0].min(), grid_points[:, 0].max())
    plt.ylim(grid_points[:, 1].min(), grid_points[:, 1].max())
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    # Corrigir espelhamento visual, se necessário
    #plt.gca().invert_xaxis()  # Descomente se o gráfico parecer refletido

    plt.show()


# ============================
# Geração de grade 2D
# ============================
def generate_grid(x_lim, y_lim, num_points=100):
    """
    Gera uma grade de pontos dentro dos limites especificados.

    Parameters:
    x_lim (tuple): Limites do eixo x (min, max).
    y_lim (tuple): Limites do eixo y (min, max).
    num_points (int): Número de pontos por eixo.

    Returns:
    np.ndarray: Array de pontos da grade no formato (N x 2).
    """
    x = np.linspace(x_lim[0], x_lim[1], num_points)
    y = np.linspace(y_lim[0], y_lim[1], num_points)
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


'''''''''
def plot_velocity_probability_map(x_prev, u, alphas, delta_t, r=0.1, res=100):
    """
    Plota o mapa de probabilidade (heatmap) do modelo de movimento por velocidade.
    """
    # Pose esperada (sem ruído)
    v, w = u
    theta = x_prev[2]
    if abs(w) > 1e-6:
        r_hat = v / w
        dx = -r_hat * np.sin(theta) + r_hat * np.sin(theta + w * delta_t)
        dy = r_hat * np.cos(theta) - r_hat * np.cos(theta + w * delta_t)
    else:
        dx = v * delta_t * np.cos(theta)
        dy = v * delta_t * np.sin(theta)
    x_exp = x_prev[0] + dx
    y_exp = x_prev[1] + dy
    theta_exp = theta + w * delta_t
    x_next_est = (x_exp, y_exp, theta_exp)

    # Geração do grid de pontos em (x, y)
    span = 1.0  # quanto ao redor da pose esperada
    x_vals = np.linspace(x_exp - span, x_exp + span, res)
    y_vals = np.linspace(y_exp - span, y_exp + span, res)

    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Mantemos theta constante na estimativa (alternativamente, variar θ também)
    for i in range(res):
        for j in range(res):
            x_test = (X[i, j], Y[i, j], theta_exp)
            Z[i, j] = prob_velocity_model(x_prev, x_test, u, alphas, delta_t)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.title("Velocity Motion Model - Probability Map")
    plt.xlabel("Position x")
    plt.ylabel("Position y")
    plt.grid(True)

    plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar(label="Probability density")

    # Robô anterior
    circle_prev = Circle((x_prev[0], x_prev[1]), r, color='green', ec='black', alpha=0.7, label='Previous Pose')
    plt.gca().add_patch(circle_prev)
    plt.arrow(x_prev[0], x_prev[1], r*np.cos(x_prev[2]), r*np.sin(x_prev[2]),
              head_width=0.03, fc='black', ec='black')

    # Pose estimada
    circle_est = Circle((x_exp, y_exp), r, color='cyan', ec='black', alpha=0.7, label='Expected Pose')
    plt.gca().add_patch(circle_est)
    plt.arrow(x_exp, y_exp, r*np.cos(theta_exp), r*np.sin(theta_exp),
              head_width=0.03, fc='magenta', ec='magenta')

    plt.plot([x_prev[0], x_exp], [x_prev[1], y_exp], 'r--', label="Expected Motion")

    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()
'''''''''

