import numpy as np
import matplotlib.pyplot as plt

from intrinsic_param import learn_intrinsic_parameters
from plot_utils import plot_combined_probabilities
from probabilities import p_hit, p_short, p_random, p_max, calc_full_probabilities

if __name__ == "__main__":
    # Dados de exemplo
    Z = np.random.normal(loc=20, scale=2, size=1000)  # 100 pontos aleatórios entre 0 e 20
    Z_exp = np.linspace(0, 40, len(Z))
    z_max = 50
    N = 10000

    # Aprender parâmetros intrínsecos
    params = learn_intrinsic_parameters(Z, Z_exp, z_max, num_iters=N)

    print("Parâmetros aprendidos:")
    print(f"z_hit: {params['z_hit']}")
    print(f"z_short: {params['z_short']}")
    print(f"z_max: {params['z_max']}")
    print(f"z_rand: {params['z_rand']}")
    print(f"sigma: {params['sigma']}")
    print(f"lambda: {params['lambda']}")
    # Plotar a combinação linear das distribuições
    z_values = np.linspace(0, z_max, 1000)
    z_exp = np.mean(Z_exp)
    plot_combined_probabilities(z_exp, z_max, params)

    



