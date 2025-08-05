import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# from calculate_prob import (    
#     prob_normal_distribution,
#     prob_triangular_distribution
# )

import math

def prob_normal_distribution(error: float, variance: float) -> float:
    """
    Returns the probability density of 'error' under a zero-mean normal distribution
    with given variance.
    """
    if variance <= 0:
        raise ValueError("Variance must be positive")
    return (1.0 / math.sqrt(2 * math.pi * variance)) * math.exp(-0.5 * (error ** 2) / variance)

def motion_model_velocity(x_t, u_t, x_prev, dt: float, alpha) -> float:
    """    
    Args:
        x_t: Estado atual [x', y', θ'] (list or tuple of floats)
        u_t: Comando de controle [v, ω] (list or tuple of floats)
        x_prev: Estado anterior [x, y, θ] (list or tuple of floats)
        dt: Passo de tempo (float)
        alpha: Parâmetros de ruído [α1, α2, α3, α4, α5, α6] (list of 6 floats)
        
    Returns:
        float: Probabilidade do movimento p(x_t | u_t, x_prev)
    """
    # unpack previous and current state
    x, y, theta = x_prev
    x_prime, y_prime, theta_prime = x_t
    v, omega = u_t

    # -----------------------------------------------------------------
    # 1. COMPUTE CIRCLE CENTER
    #    use (x' - x) e (y' - y) for both numerator and denominator
    #    and midpoint orientation for better robustness
    # -----------------------------------------------------------------
    theta_mid = 0.5 * (theta + theta_prime)
    dx = x_prime - x
    dy = y_prime - y

    numer =  dx * math.cos(theta_mid) + dy * math.sin(theta_mid)
    denom =  dy * math.cos(theta_mid) - dx * math.sin(theta_mid)

    # dynamic threshold: relative to distance moved
    dist = math.hypot(dx, dy)
    thresh = 1e-6 * max(1.0, dist)

    # straight-line motion (denom ≈ 0)
    if abs(denom) < thresh:
        # -----------------------------------------------------------------
        # 1a. Straight motion: center at infinity
        # -----------------------------------------------------------------
        v_hat = dist / dt
        omega_hat = 0.0
        gamma = (theta_prime - theta) / dt  # no correction needed since omega_hat=0
    else:
        # -----------------------------------------------------------------
        # 1b. Circular motion
        # -----------------------------------------------------------------
        mu = 0.5 * numer / denom
        # center of rotation
        x_star = 0.5 * (x + x_prime) + mu * (y - y_prime)
        y_star = 0.5 * (y + y_prime) + mu * (x_prime - x)

        # -----------------------------------------------------------------
        # 2. RADIUS AND ESTIMATED VELOCITIES
        # -----------------------------------------------------------------
        r_star = math.hypot(x - x_star, y - y_star)

        # delta angle along arc
        theta1 = math.atan2(y - y_star, x - x_star)
        theta2 = math.atan2(y_prime - y_star, x_prime - x_star)
        delta_theta = theta2 - theta1
        # normalize to [-π, π]
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))

        omega_hat = delta_theta / dt
        v_hat = omega_hat * r_star

        # correction term
        gamma = (theta_prime - theta) / dt - omega_hat

    # -----------------------------------------------------------------
    # 3. COMPUTE NOISE VARIANCES
    # -----------------------------------------------------------------
    var1 = alpha[0] * v**2 + alpha[1] * omega**2
    var2 = alpha[2] * v**2 + alpha[3] * omega**2
    var3 = alpha[4] * v**2 + alpha[5] * omega**2

    # -----------------------------------------------------------------
    # 4. COMPUTE INDIVIDUAL PROBABILITIES
    # -----------------------------------------------------------------
    p1 = prob_normal_distribution(v - v_hat, var1)        # linear velocity error
    p2 = prob_normal_distribution(omega - omega_hat, var2)  # angular velocity error
    p3 = prob_normal_distribution(gamma, var3)             # final orientation error

    # assume independence
    return p1 * p2 * p3



# Parâmetros de exemplo
alpha = [0.05, 0.05, 1.8, 1.8, 2.5, 2.5]  # Parâmetros de ruído
alpha = [0.5, 0.5, 0.1, 0.1, 0.5, 0.5]
dt = 0.5                                 # Passo de tempo
# Estado anterior (t-1)
x_prev = np.array([0.0, 0.0, 0.0])            # [x, y, θ]

# Comando de controle (u_t)
u_t = np.array([3.0, 0.5])                    # [v, ω]

expected_angle = u_t[1] * dt
expected_x = (u_t[0]/u_t[1]) * math.sin(expected_angle)
expected_y = (u_t[0]/u_t[1]) * (1 - math.cos(expected_angle))

# Estado atual (t) - ligeiramente diferente do esperado
x_t = [expected_x, expected_y, expected_angle] 

# Configuração da grade
x_grid = np.linspace(-2.0, 2.0, 400)
y_grid = np.linspace(-2.0, 2.0, 400)
# prob_grid = np.zeros((len(x_grid), len(y_grid)))
prob_grid = np.zeros((len(y_grid), len(x_grid)))

# Preenche a grade de probabilidades
for i, x in enumerate(x_grid):
    for j, y in enumerate(y_grid):
        x_candidate = [x, y, 0.05]  # θ fixo para exemplo
        prob_grid[j, i] = motion_model_velocity(
                x_candidate, u_t, x_prev, dt, alpha
            )
        # if y > 0:
        #     prob_grid[j, i] = motion_model_velocity(
        #         x_candidate, u_t, x_prev, dt, alpha
        #     )
        # else:
        #     prob_grid[j, -i] = motion_model_velocity(
        #         x_candidate, u_t, x_prev, dt, alpha
        #     )
prob_min = np.min(prob_grid)
prob_max = np.max(prob_grid)

if prob_max - prob_min > 1e-10:
    prob_grid_normalized = (prob_grid - prob_min) / (prob_max - prob_min)
else:
    prob_grid_normalized = prob_grid
# Plotagem
#prob_grid_normalized = prob_grid
plt.figure(figsize=(10, 8))
plt.imshow(prob_grid_normalized, 
           extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
           origin='lower', 
           cmap='viridis')
plt.colorbar(label='Probabilidade')

r = 0.08

circle_prev = Circle((x_prev[0], x_prev[1]), r, color='green', ec='black', alpha=0.7, label='Previous Robot')
plt.gca().add_patch(circle_prev)
plt.arrow(x_prev[0], x_prev[1], r * np.cos(x_prev[2]), r * np.sin(x_prev[2]),
        head_width=0.03, fc='black', ec='black')

circle_est = Circle((x_t[0], x_t[1]), r, color='cyan', ec='black', alpha=0.7, label='Expected Robot')
plt.gca().add_patch(circle_est)
plt.arrow(x_t[0], x_t[1], r * np.cos(x_t[2]), r * np.sin(x_t[2]),
        head_width=0.03, fc='magenta', ec='magenta')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Densidade de Probabilidade do Motion Model')

theta_vals = np.linspace(0, expected_angle, 50)
x_vals = (u_t[0]/u_t[1]) * np.sin(theta_vals)
y_vals = (u_t[0]/u_t[1]) * (1 - np.cos(theta_vals))
plt.plot(x_vals, y_vals, 'r--', alpha=0.7, label='Trajetória Esperada')

plt.legend()
plt.show()
