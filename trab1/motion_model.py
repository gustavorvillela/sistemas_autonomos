import numpy as np
import math
from generate_samples import sample_normal_distribution
from calculate_prob import *

def sample_motion_model_odometry(x_prev, odo_prev, odo_curr, alphas, num_samples=1000):
    """Gera partículas de acordo com o modelo de movimento por odometria"""
    
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

    particle = [x_sample, y_sample, theta_sample]

    return np.array(particle)


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
    
    # Normalização: garante soma total igual a 1
    total = np.sum(probabilities)
    if total > 0:
        probabilities /= total

    return probabilities


def normalize_angle(theta):
    """Normaliza um ângulo para o intervalo [-pi, pi]"""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def sample_motion_model_velocity(x_prev, u, alphas, delta_t):
    """
    Gera partículas para o modelo de movimento por velocidade com ruído.
    x_prev: pose anterior (x, y, theta)
    u: controle (v, w)
    alphas: ruído [α1...α6]
    """
    v, w = u
    x, y, theta = x_prev
    
    # Adiciona ruído aos comandos de controle
    v_hat = v + sample_normal_distribution(alphas[0] * abs(v)**2 + alphas[1] * abs(w)**2)
    w_hat = w + sample_normal_distribution(alphas[2] * abs(v)**2 + alphas[3] * abs(w)**2)
    gamma_hat = sample_normal_distribution(alphas[4] * abs(v)**2 + alphas[5] * abs(w)**2)

    theta_hat = theta + w_hat * delta_t 

    if abs(w_hat) > 1e-6:
        r_hat = v_hat / w_hat
        dx = -r_hat * np.sin(theta) + r_hat * np.sin(theta_hat)
        dy = r_hat * np.cos(theta) - r_hat * np.cos(theta_hat)
    else:
        dx = v_hat * delta_t * np.cos(theta)
        dy = v_hat * delta_t * np.sin(theta)

    dtheta = w_hat * delta_t + gamma_hat * delta_t

    x_new = x + dx
    y_new = y + dy
    theta_new = normalize_angle(theta + dtheta)

    particle = (x_new, y_new, theta_new)
    
    return particle

def deterministic_motion_model_velocity(x_prev, u, delta_t):
    """
    Gera partículas para o modelo de movimento por velocidade com ruído.
    x_prev: pose anterior (x, y, theta)
    u: controle (v, w)
    alphas: ruído [α1...α6]
    """
    v, w = u
    x, y, theta = x_prev
    
    v_hat = v 
    w_hat = w 

    theta_hat = theta + w_hat * delta_t 

    r_hat = v_hat / w_hat
    dx = -r_hat * np.sin(theta) + r_hat * np.sin(theta_hat)
    dy = r_hat * np.cos(theta) - r_hat * np.cos(theta_hat)


    dtheta = w_hat * delta_t

    x_new = x + dx
    y_new = y + dy
    theta_new = normalize_angle(theta + dtheta)

    particle = (x_new, y_new, theta_new)
    
    return particle


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
    Returns the probability of ending at x_curr given:
    - x_prev = [x, y, theta]   (pose at time t-1)
    - u = [v, w]               (control at t-1)
    - alphas = motion noise coefficients
    - delta_t = time interval
    """

    x, y, theta = x_prev
    xc, yc, thetac = x_curr
    v, w = u

    # Estimate rotation center
    dx = xc - x
    dy = yc - y

    num = dx * np.cos(theta) + dy * np.sin(theta)
    denom = dy * np.cos(theta) - dx * np.sin(theta)

    if abs(denom) < 1e-6:
        mu = 0.0
    else:
        mu = 0.5 * num / denom

    x_star = 0.5 * (x + xc) + mu * dy
    y_star = 0.5 * (y + yc) + mu * -dx

    r_star = np.sqrt((x - x_star)**2 +  (y - y_star)**2)

    # Vectors from rotation center to poses
    v1 = np.array([x - x_star, y - y_star])
    v2 = np.array([xc - x_star, yc - y_star])

    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    delta_theta = normalize_angle(angle1 - angle2)

    w_hat =  delta_theta / delta_t

    # Use dot product to determine if motion was forward or backward
    heading = np.array([np.cos(theta), np.sin(theta)])
    displacement = np.array([dx, dy])
    v_sign = np.sign(np.dot(heading, displacement)) or 1.0
    v_hat = v_sign * abs(w_hat * r_star)

    # Estimate rotation change unrelated to arc
    gamma_hat = normalize_angle(thetac - theta) / delta_t - w_hat

    # Variances based on noise
    var_v = alphas[0] * abs(v) + alphas[1] * abs(w)
    var_w = alphas[2] * abs(v) + alphas[3] * abs(w)
    var_gamma = alphas[4] * abs(v) + alphas[5] * abs(w)

    # Errors between commanded and estimated
    v_err = v - v_hat
    w_err = w - w_hat

    # Final probability as product of Gaussians
    p1 = prob_normal_distribution(v_err, var_v)
    p2 = prob_normal_distribution(w_err, var_w)
    p3 = prob_normal_distribution(gamma_hat, var_gamma)

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
