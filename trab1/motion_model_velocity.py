import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from calculate_prob import (    
    prob_normal_distribution,
    prob_triangular_distribution
)

def motion_model_velocity(x_t, u_t, x_prev, dt:float, alpha) -> float:
    """    
    Args:
        x_t: Estado atual [x', y', θ'] (numpy array ou lista)
        u_t: Comando de controle [v, ω] (numpy array ou lista)
        x_prev: Estado anterior [x, y, θ] (numpy array ou lista)
        dt: Passo de tempo (float)
        alpha: Parâmetros de ruído [α1, α2, α3, α4, α5, α6] (lista)
        
    Returns:
        float: Probabilidade do movimento p(x_t | u_t, x_prev)
    """
    # Desempacota os estados e controle
    x, y, theta = x_prev
    x_prime, y_prime, theta_prime = x_t
    v, omega = u_t
    
    # =====================================================================
    # 1. Cálculo do centro do arco circular (x*, y*)
    # =====================================================================
    numer:float = (x - x_prime) * math.cos(theta) + (y - y_prime) * math.sin(theta)
    denom:float = (y - y_prime) * math.cos(theta) - (x  - x_prime) * math.sin(theta)
    
    # Trata caso singular (movimento retilíneo)
    if math.isclose(denom, 0, abs_tol=1e-6):
        # Movimento retilíneo - centro no infinito
        dist = math.sqrt((x_prime - x)**2 + (y_prime - y)**2)
        v_hat = dist / dt
        omega_hat = 0.0
        gamma = (theta_prime - theta) / dt
    else:
        # Movimento circular
        mu = (0.5 * numer) / denom
        # Calcula o centro do arco
        x_star = (x + x_prime)/2.0 + mu*(y - y_prime)
        y_star = (y + y_prime)/2.0 + mu*(x_prime - x)
        
        # =================================================================
        # 2. Cálculo do raio e velocidades estimadas
        # =================================================================
        # Raio do arco
        r_star = math.sqrt((x - x_star)**2 + (y - y_star)**2)
        
        # Diferença angular
        delta_theta = math.atan2(y_prime - y_star, x_prime - x_star) - math.atan2(y - y_star, x - x_star)
        
        # Normaliza o ângulo entre [-π, π]
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))
        
        # Velocidades estimadas
        omega_hat = delta_theta / dt
        v_hat = omega_hat * r_star
        
        # Correção da orientação final
        gamma = (theta_prime - theta)/dt - omega_hat
    
    # =====================================================================
    # 3. Cálculo das probabilidades dos erros
    # =====================================================================
    # Calcula variâncias
    var1 = alpha[0]*v**2 + alpha[1]*omega**2
    var2 = alpha[2]*v**2 + alpha[3]*omega**2
    var3 = alpha[4]*v**2 + alpha[5]*omega**2
    
    # Probabilidades individuais
    p1 = prob_normal_distribution(v - v_hat, var1)        # Erro velocidade linear
    p2 = prob_normal_distribution(omega - omega_hat, var2) # Erro velocidade angular
    p3 = prob_normal_distribution(gamma, var3)             # Erro orientação final
    
    # Probabilidade final (assume independência)
    return p1 * p2 * p3


def motion_model_velocity2(x_t, u_t, x_prev, dt, alpha, debug=False):
    """
    Implementação fiel do modelo de movimento baseado em velocidade
    
    Args:
        x_t: Estado atual [x', y', θ']
        u_t: Comando de controle [v, ω]
        x_prev: Estado anterior [x, y, θ]
        dt: Passo de tempo
        alpha: Parâmetros de ruído [α1-α6]
        
    Returns:
        Probabilidade do movimento
    """
    # Desempacota estados
    x, y, theta = x_prev
    x_prime, y_prime, theta_prime = x_t
    v, omega = u_t
    
    # 1. Cálculo do centro do arco circular (CORRETO)
    numer = (x - x_prime) * math.cos(theta) + (y - y_prime) * math.sin(theta)
    denom = (y - y_prime) * math.cos(theta) - (x - x_prime) * math.sin(theta)
    
    # Trata caso singular (movimento retilíneo)
    if abs(denom) < 1e-6:
        dist = math.sqrt((x_prime - x)**2 + (y_prime - y)**2)
        v_hat = dist / dt
        omega_hat = 0.0
        gamma = (theta_prime - theta) / dt
        if debug:
            print(f"Movimento retilíneo: v_hat={v_hat:.4f}, gamma={gamma:.4f}")
    else:
        # 2. Cálculo do centro e raio (CORRETO)
        mu = 0.5 * numer / denom
        x_star = (x + x_prime)/2.0 + mu*(y - y_prime)
        y_star = (y + y_prime)/2.0 + mu*(x_prime - x)
        r_star = math.sqrt((x - x_star)**2 + (y - y_star)**2)
        
        # 3. Cálculo do ângulo (CORREÇÃO CRÍTICA - fiel ao algoritmo)
        # Algoritmo original: Δθ = atan2(y'-y, x'-x) - atan2(y-y*,x-x*)
        angle1 = math.atan2(y_prime - y, x_prime - x)  # Vetor deslocamento: anterior → atual
        angle2 = math.atan2(y - y_star, x - x_star)    # Vetor centro → anterior
        delta_theta = angle1 - angle2
        
        # Normalização angular robusta
        delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))
        
        # 4. Estimativa de velocidades (CORRETO)
        omega_hat = delta_theta / dt
        v_hat = omega_hat * r_star
        gamma = (theta_prime - theta)/dt - omega_hat
        
        if debug:
            print(f"Centro do arco: ({x_star:.4f}, {y_star:.4f})")
            print(f"Raio: {r_star:.4f} m, Δθ: {math.degrees(delta_theta):.2f}°")
            print(f"v_hat={v_hat:.4f} m/s, ω_hat={omega_hat:.4f} rad/s, γ={gamma:.6f}")

    # 5. Cálculo de variâncias (CORRETO)
    var1 = alpha[0] * v**2 + alpha[1] * omega**2
    var2 = alpha[2] * v**2 + alpha[3] * omega**2
    var3 = alpha[4] * v**2 + alpha[5] * omega**2
    
    # 6. Cálculo de probabilidades (CORRETO)
    p1 = prob_normal_distribution(v - v_hat, var1)
    p2 = prob_normal_distribution(omega - omega_hat, var2)
    p3 = prob_normal_distribution(gamma, var3)
    
    if debug:
        print(f"Variâncias: v={var1:.6f}, ω={var2:.6f}, γ={var3:.6f}")
        print(f"Probabilidades: p1={p1:.6f}, p2={p2:.6f}, p3={p3:.6f}")
    
    # 7. Probabilidade final
    probability = p1 * p2 * p3
    if debug:
        print(f"Probabilidade final: {probability:.10f}")
    
    return probability
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
prob_grid = np.zeros((len(x_grid), len(y_grid)))

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
