import numpy as np
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
