import math
import numpy as np
import matplotlib.pyplot as plt

from calculate_prob import (    
    prob_normal_distribution,
    prob_triangular_distribution
)

def normal_prob(error, variance):
    """
    Calcula a probabilidade usando distribuição normal (Gaussiana)
    
    Args:
        error: Diferença entre o valor real e o estimado
        variance: Variância da distribuição
        
    Returns:
        float: Probabilidade do erro dado a variância
    """
    if variance <= 0:
        # Caso a variância seja zero, retorna 1 se erro for zero, senão 0
        return 1.0 if math.isclose(error, 0, abs_tol=1e-9) else 0.0
    return (1.0 / math.sqrt(2 * math.pi * variance)) * math.exp(-(error**2) / (2 * variance))

def motion_model_velocity(x_t, u_t, x_prev, dt, alpha):
    """
    Modelo de movimento baseado em velocidade para robótica móvel
    
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
    numer = (x - x_prime) * math.cos(theta) + (y - y_prime) * math.sin(theta)
    denom = (y_prime - y) * math.cos(theta) - (x_prime - x) * math.sin(theta)
    
    # Trata caso singular (movimento retilíneo)
    if math.isclose(denom, 0, abs_tol=1e-6):
        # Movimento retilíneo - centro no infinito
        dist = math.sqrt((x_prime - x)**2 + (y_prime - y)**2)
        v_hat = dist / dt
        omega_hat = 0.0
        gamma = (theta_prime - theta) / dt
    else:
        # Movimento circular
        mu = 0.5 * numer / denom
        
        # Calcula o centro do arco
        x_star = (x + x_prime)/2.0 + mu*(y - y_prime)
        y_star = (y + y_prime)/2.0 + mu*(x_prime - x)
        
        # =================================================================
        # 2. Cálculo do raio e velocidades estimadas
        # =================================================================
        # Raio do arco
        r_star = math.sqrt((x - x_star)**2 + (y - y_star)**2)
        
        # Diferença angular
        delta_theta = math.atan2(y_prime - y, x_prime - x) - math.atan2(y - y_star, x - x_star)
        
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
    p1 = normal_prob(v - v_hat, var1)        # Erro velocidade linear
    p2 = normal_prob(omega - omega_hat, var2) # Erro velocidade angular
    p3 = normal_prob(gamma, var3)             # Erro orientação final
    
    # Probabilidade final (assume independência)
    return p1 * p2 * p3


# Parâmetros de exemplo
alpha = [0.1, 0.1, 0.01, 0.01, 0.01, 0.01]  # Parâmetros de ruído
dt = 0.1                                    # Passo de tempo
# Estado anterior (t-1)
x_prev = np.array([0.0, 0.0, 0.0])            # [x, y, θ]

# Comando de controle (u_t)
u_t = np.array([1.0, 0.5])                    # [v, ω]

x_t = np.array([0.097, 0.005, 0.049])

# Configuração da grade
x_grid = np.linspace(-0.5, 0.5, 100)
y_grid = np.linspace(-0.5, 0.5, 100)
prob_grid = np.zeros((len(x_grid), len(y_grid)))

# Preenche a grade de probabilidades
for i, x in enumerate(x_grid):
    for j, y in enumerate(y_grid):
        x_candidate = [x, y, 0.05]  # θ fixo para exemplo
        prob_grid[j, i] = motion_model_velocity(
            x_candidate, u_t, x_prev, dt, alpha
        )

# Plotagem
# plt.figure(figsize=(10, 8))
# plt.imshow(prob_grid, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], origin='lower')
# plt.colorbar(label='Probabilidade')
# plt.scatter(x_prev[0], x_prev[1], c='black', s=100, label='Estado Anterior')
# plt.scatter(x_t[0], x_t[1], c='red', s=100, label='Estado Atual')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Densidade de Probabilidade do Motion Model')
# plt.legend()
# plt.show()

for i in prob_grid:
  print(i*100000)