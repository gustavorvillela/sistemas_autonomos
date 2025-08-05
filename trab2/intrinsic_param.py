import numpy as np
from probabilities import p_hit, p_short, p_random, p_max
import math


def learn_intrinsic_parameters(Z, Z_exp, z_max, num_iters=20, z_init=(0.25, 0.25, 0.25, 0.25)):
    """
    Função para aprender os parâmetros intrínsecos do modelo de probabilidade.
    """
    N = len(Z)

    # Inicialização dos parâmetros
    sigma = 1.0
    lambd = 0.01
    z_hit, z_short, z_max_w, z_rand = z_init
    print(f"Initial parameters: z_hit={z_hit}, z_short={z_short}, z_max={z_max_w}, z_rand={z_rand}, sigma={sigma}, lambda={lambd}")

    for _ in range(num_iters):
        e_hit = np.zeros(N)
        e_short = np.zeros(N)
        e_max = np.zeros(N)
        e_rand = np.zeros(N)

        for i in range(N):
            z = Z[i]
            z_exp = Z_exp[i]

            # Cálculo de eta
            p1 = p_hit(z, z_exp, sigma)
            p2 = p_short(z, z_exp, lambd)
            p3 = p_max(z, z_max)
            p4 = p_random(z, z_max)

            eta_inv = z_hit * p1 + z_short * p2 + z_max_w * p3 + z_rand * p4
            eta = 1.0 / (eta_inv + 1e-12)

            e_hit[i] = eta * p1   #* z_hit
            e_short[i] = eta * p2 #* z_short
            e_max[i] = eta * p3   #* z_max_w 
            e_rand[i] = eta * p4  #* z_rand

        # Atualização dos pesos
        z_hit = np.sum(e_hit) / N
        z_short = np.sum(e_short) / N
        z_max_w = np.sum(e_max) / N
        z_rand = np.sum(e_rand) / N

        # Atualizar sigma_hit
        sigma_num = np.sum(e_hit * ((Z - Z_exp) ** 2))
        sigma_den = np.sum(e_hit) + 1e-12
        sigma = math.sqrt(sigma_num / sigma_den)

        # Atualizar lambda_short
        lambda_num = np.sum(e_short)
        lambda_den = np.sum(e_short * Z) + 1e-12
        lambd = lambda_num / lambda_den

    return {
        "z_hit": z_hit,
        "z_short": z_short,
        "z_max": z_max_w,
        "z_rand": z_rand,
        "sigma": sigma,
        "lambda": lambd
    }
