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