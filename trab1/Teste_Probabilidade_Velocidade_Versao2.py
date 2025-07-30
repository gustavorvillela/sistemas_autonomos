import numpy as np
from motion_model import velocity_prob_grid, generate_motion_model_velocity_particles
from plot_utils import (
    plot_velocity_particles,
    generate_grid,
    plot_prob_velocity
)

if __name__ == "__main__":
    # Estado anterior e estimado (esperado)
    x_prev = (0.0, 0.0, 0.0)
    x_real = (1.0, 0.5, np.pi / 8)

    # Comando de velocidade (v, w)
    u = (1.0, 0.5)

    # Parâmetros do modelo
    alphas = [0.1]*6
    delta_t = 1.0

    # Gerar partículas a partir do modelo de movimento por velocidade
    particles = generate_motion_model_velocity_particles(x_prev, u, alphas, delta_t, num_samples=1000)
    plot_velocity_particles(x_prev, x_real, particles, r=0.1)

    # Gerar grade e calcular probabilidades para a nuvem
    grid = generate_grid((-2, 2), (-2, 2), num_points=100)
    probabilities = velocity_prob_grid(grid, x_prev, x_real[2], u, alphas, delta_t)

    plot_prob_velocity(grid, probabilities, x_prev, x_real, r=0.1, title="Velocity Probability Distribution")
