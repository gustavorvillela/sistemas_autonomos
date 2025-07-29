import numpy as np
from motion_model import sample_motion_model_velocity
from plot_utils import plot_velocity_particles


if __name__ == "__main__":
    # Pose anterior
    x_prev = (0.0, 0.0, np.pi/6)

    # Comando (velocidade linear e angular)
    u = (1.0, 0.5)

    # Estimativa esperada da pose (sem ruído, cálculo direto)
    delta_t = 1.0
    v, w = u
    theta = x_prev[2]

    # Trajetória sem ruído
    if abs(w) > 1e-6:
        r = v / w
        dx = -r * np.sin(theta) + r * np.sin(theta + w * delta_t)
        dy = r * np.cos(theta) - r * np.cos(theta + w * delta_t)
    else:
        dx = v * delta_t * np.cos(theta)
        dy = v * delta_t * np.sin(theta)

    x_est = x_prev[0] + dx
    y_est = x_prev[1] + dy
    theta_est = theta + w * delta_t
    x_next_est = (x_est, y_est, theta_est)

    alphas = [0.1]*6
    particles = sample_motion_model_velocity(x_prev, u, alphas, delta_t, num_samples=2000)

    plot_velocity_particles(x_prev, x_next_est, particles, r=0.1)
