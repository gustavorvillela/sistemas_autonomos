import numpy as np
from generate_samples import sample_normal_distribution
from calculate_prob import *

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


def generate_motion_model_particles(x_prev, odo_prev, odo_curr, alphas, num_samples=1000):
    """
    Gera partículas de acordo com o modelo de movimento por odometria.

    Parameters:
    x_prev (tuple): Posição e orientação anterior do robô (x, y, theta).
    odo_prev (tuple): Odometria anterior (xb, yb, thetab).
    odo_curr (tuple): Odometria atual (xbp, ybp, thetabp).
    alphas (list): Parâmetros de ruído do modelo de movimento.
    num_samples (int): Número de partículas a serem geradas.

    Returns:
    np.ndarray: Partículas geradas pelo modelo de movimento.
    """

    particles = [sample_motion_model_odometry(x_prev, odo_prev, odo_curr, alphas) for _ in range(num_samples)]

    return np.array(particles)


def motion_model_odometry(x_prev, x_curr, odo_prev, odo_curr, alphas):
    """
    Calcula a probabilidade de acordo com o modelo de movimento por odometria.

    Parameters:
    x_prev (tuple): Posição e orientação anterior do robô (x, y, theta).
    x_curr (tuple): Posição e orientação atual do robô (x, y, theta).
    odo_prev (tuple): Odometria anterior (xb, yb, thetab).
    odo_curr (tuple): Odometria atual (xbp, ybp, thetabp).
    alphas (list): Parâmetros de ruído do modelo de movimento.

    Returns:
    np.ndarray: Partículas geradas pelo modelo de movimento.
    """

    x, y, theta = x_prev
    xc, yc, thetac = x_curr
    xb, yb, thetab = odo_prev
    xbp, ybp, thetabp = odo_curr

    delta_rot1 = np.arctan2(ybp - yb, xbp - xb) - thetab
    delta_trans = np.sqrt((xbp - xb) ** 2 + (ybp - yb) ** 2)
    delta_rot2 = thetabp - thetab - delta_rot1

    delta_rot1_hat = np.arctan2(yc - y, xc - x) - theta
    delta_trans_hat = np.sqrt((xc - x) ** 2 + (yc - y) ** 2)
    delta_rot2_hat = thetac - theta - delta_rot1_hat

    p1 = prob_normal_distribution(delta_rot1 - delta_rot1_hat, alphas[0] * abs(delta_rot1) + alphas[1] * delta_trans)
    p2 = prob_normal_distribution(delta_trans - delta_trans_hat,
                                  alphas[2] * abs(delta_rot1) + alphas[3] * (abs(delta_rot1) + abs(delta_rot2)))
    p3 = prob_normal_distribution(delta_rot2 - delta_rot2_hat, alphas[4] * abs(delta_rot2) + alphas[5] * delta_trans)

    return p1 * p2 * p3


def odometry_prob_grid(grid, x_prev, angle_cur, odo_prev, odo_curr, alphas):
    """
    Calcula a probabilidade de cada ponto da grade de acordo com o modelo de movimento por odometria.

    Parameters:
    grid (np.ndarray): Grade de pontos (Nx2).
    x_prev (tuple): Posição e orientação anterior do robô (x, y, theta).
    odo_prev (tuple): Odometria anterior (xb, yb, thetab).
    odo_curr (tuple): Odometria atual (xbp, ybp, thetabp).
    alphas (list): Parâmetros de ruído do modelo de movimento.

    Returns:
    np.ndarray: Probabilidades para cada ponto da grade.
    """

    probabilities = np.array(
        [motion_model_odometry(x_prev, (point[0], point[1], angle_cur), odo_prev, odo_curr, alphas) for point in grid])

    return probabilities

''''''''''
def sample_motion_model_velocity(x_prev, u, alphas, delta_t, num_samples=1000):
    """
    Gera partículas para o modelo de movimento por velocidade com ruído.
    x_prev: pose anterior (x, y, theta)
    u: controle (v, w)
    alphas: ruído [α1...α6]
    """
    v, w = u
    x, y, theta = x_prev
    particles = []

    for _ in range(num_samples):
        # Adiciona ruído aos comandos de controle
        v_hat = v + sample_normal_distribution(alphas[0] * v**2 + alphas[1] * w**2)
        w_hat = w + sample_normal_distribution(alphas[2] * v**2 + alphas[3] * w**2)
        gamma_hat = sample_normal_distribution(alphas[4] * v**2 + alphas[5] * w**2)

        if abs(w_hat) > 1e-6:
            r_hat = v_hat / w_hat
            dx = -r_hat * np.sin(theta) + r_hat * np.sin(theta + w_hat * delta_t)
            dy = r_hat * np.cos(theta) - r_hat * np.cos(theta + w_hat * delta_t)
        else:
            # Movimento quase retilíneo
            dx = v_hat * delta_t * np.cos(theta)
            dy = v_hat * delta_t * np.sin(theta)

        dtheta = w_hat * delta_t + gamma_hat * delta_t

        x_new = x + dx
        y_new = y + dy
        theta_new = theta + dtheta

        particles.append((x_new, y_new, theta_new))

    return np.array(particles)


def generate_motion_model_velocity_particles(x_prev, u, alphas, delta_t, num_samples=1000):
    """
    Gera partículas de acordo com o modelo de movimento por velocidade.

    Parameters:
    x_prev (tuple): Posição e orientação anterior do robô (x, y, theta).
    u (tuple): Comando de velocidade (v, w).
    alphas (list): Parâmetros de ruído do modelo de movimento.
    delta_t (float): Intervalo de tempo.
    num_samples (int): Número de partículas a serem geradas.

    Returns:
    np.ndarray: Partículas geradas pelo modelo de movimento por velocidade.
    """
    particles = [sample_motion_model_velocity(x_prev, u, alphas, delta_t) for _ in range(num_samples)]
    return np.array(particles)


def motion_model_velocity(x_prev, x_curr, u, alphas, delta_t):
    """
    Calcula a probabilidade de uma pose x_curr ter sido gerada a partir de x_prev com controle u
    segundo o modelo de movimento por velocidade.

    Parameters:
    x_prev (tuple): Pose anterior (x, y, theta).
    x_curr (tuple): Pose atual (x', y', theta').
    u (tuple): Comando de controle (v, w).
    alphas (list): Parâmetros de ruído do modelo.
    delta_t (float): Intervalo de tempo.

    Returns:
    float: Probabilidade segundo o modelo de movimento.
    """
    x, y, theta = x_prev
    x_p, y_p, theta_p = x_curr
    v, w = u
    α1, α2, α3, α4, α5, α6 = alphas

    mu = 0.5 * ((x - x_p) * np.cos(theta) + (y - y_p) * np.sin(theta))
    x_star = 0.5 * (x + x_p) + mu * (y - y_p)
    y_star = 0.5 * (y + y_p) + mu * (x_p - x)

    r_star = np.sqrt((x - x_star)**2 + (y - y_star)**2)
    delta_theta = np.arctan2(y_p - y_star, x_p - x_star) - np.arctan2(y - y_star, x - x_star)

    w_hat = delta_theta / delta_t
    v_hat = w_hat * r_star
    gamma_hat = (theta_p - theta) / delta_t - w_hat

    p1 = prob_normal_distribution(v - v_hat, α1 * v**2 + α2 * w**2)
    p2 = prob_normal_distribution(w - w_hat, α3 * v**2 + α4 * w**2)
    p3 = prob_normal_distribution(gamma_hat, α5 * v**2 + α6 * w**2)

    return p1 * p2 * p3


def velocity_prob_grid(grid, x_prev, angle_cur, u, alphas, delta_t):
    """
    Calcula a probabilidade de cada ponto da grade de acordo com o modelo de movimento por velocidade.

    Parameters:
    grid (np.ndarray): Grade de pontos (Nx2), com coordenadas (x, y).
    x_prev (tuple): Posição e orientação anterior do robô (x, y, theta).
    angle_cur (float): Ângulo fixo (theta) para cada ponto.
    u (tuple): Comando de controle (v, w).
    alphas (list): Parâmetros de ruído do modelo de movimento.
    delta_t (float): Intervalo de tempo.

    Returns:
    np.ndarray: Probabilidades para cada ponto da grade.
    """
    probabilities = np.array([
        motion_model_velocity(x_prev, (point[0], point[1], angle_cur), u, alphas, delta_t)
        for point in grid
    ])
    return probabilities
'''''''''

def normalize_angle(theta):
    """Normaliza um ângulo para o intervalo [-pi, pi]"""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def sample_motion_model_velocity(x_prev, u, alphas, delta_t, num_samples=1000):
    """
    Gera partículas para o modelo de movimento por velocidade com ruído.
    x_prev: pose anterior (x, y, theta)
    u: controle (v, w)
    alphas: ruído [α1...α6]
    """
    v, w = u
    x, y, theta = x_prev
    particles = []

    for _ in range(num_samples):
        # Adiciona ruído aos comandos de controle
        v_hat = v + sample_normal_distribution(alphas[0] * abs(v)**2 + alphas[1] * abs(w)**2)
        w_hat = w + sample_normal_distribution(alphas[2] * abs(v)**2 + alphas[3] * abs(w)**2)
        gamma_hat = sample_normal_distribution(alphas[4] * abs(v)**2 + alphas[5] * abs(w)**2)

        if abs(w_hat) > 1e-6:
            r_hat = v_hat / w_hat
            dx = -r_hat * np.sin(theta) + r_hat * np.sin(theta + w_hat * delta_t)
            dy = r_hat * np.cos(theta) - r_hat * np.cos(theta + w_hat * delta_t)
        else:
            dx = v_hat * delta_t * np.cos(theta)
            dy = v_hat * delta_t * np.sin(theta)

        dtheta = w_hat * delta_t + gamma_hat * delta_t

        x_new = x + dx
        y_new = y + dy
        theta_new = normalize_angle(theta + dtheta)

        particles.append((x_new, y_new, theta_new))

    return np.array(particles)


def generate_motion_model_velocity_particles(x_prev, u, alphas, delta_t, num_samples=1000):
    """
    Gera partículas de acordo com o modelo de movimento por velocidade.

    Parameters:
    x_prev (tuple): Posição e orientação anterior do robô (x, y, theta).
    u (tuple): Comando de velocidade (v, w).
    alphas (list): Parâmetros de ruído do modelo de movimento.
    delta_t (float): Intervalo de tempo.
    num_samples (int): Número de partículas a serem geradas.

    Returns:
    np.ndarray: Partículas geradas pelo modelo de movimento por velocidade.
    """
    particles = [sample_motion_model_velocity(x_prev, u, alphas, delta_t) for _ in range(num_samples)]
    return np.array(particles)



def prob_velocity_model(x_prev, x_curr, u, alphas, delta_t):
    """
    Calcula a probabilidade da pose x_curr dado o modelo de movimento por velocidade.
    """
    x, y, theta = x_prev
    xc, yc, thetac = x_curr
    v, w = u

    if abs(w) > 1e-6:
        r = v / w
        dx = -r * np.sin(theta) + r * np.sin(theta + w * delta_t)
        dy = r * np.cos(theta) - r * np.cos(theta + w * delta_t)
    else:
        dx = v * delta_t * np.cos(theta)
        dy = v * delta_t * np.sin(theta)

    x_exp = x + dx
    y_exp = y + dy
    theta_exp = normalize_angle(theta + w * delta_t)

    # Erros entre a pose estimada e a atual
    dx_err = xc - x_exp
    dy_err = yc - y_exp
    dtheta_err = normalize_angle(thetac - theta_exp)

    # Cálculo das variâncias
    var_x = alphas[0] * abs(v)**2 + alphas[1] * abs(w)**2
    var_y = alphas[2] * abs(v)**2 + alphas[3] * abs(w)**2
    var_theta = alphas[4] * abs(v)**2 + alphas[5] * abs(w)**2

    p1 = prob_normal_distribution(dx_err, var_x)
    p2 = prob_normal_distribution(dy_err, var_y)
    p3 = prob_normal_distribution(dtheta_err, var_theta)

    return p1 * p2 * p3

def velocity_prob_grid(grid, x_prev, angle_cur, u, alphas, delta_t):
    """
    Calcula a probabilidade de cada ponto da grade de acordo com o modelo de velocidade.
    """
    probabilities = np.array([
        prob_velocity_model(x_prev, (point[0], point[1], angle_cur), u, alphas, delta_t)
        for point in grid
    ])

    # Normalização: garante soma total igual a 1
    total = np.sum(probabilities)
    if total > 0:
        probabilities /= total

    return probabilities
