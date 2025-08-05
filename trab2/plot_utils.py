import numpy as np
import matplotlib.pyplot as plt
from probabilities import p_hit, p_short, p_random, p_max, calc_full_probabilities

def plot_probabilities(z_exp, z_max, sigma, lbd):
    """
    Plota as distribuições de probabilidade para os parâmetros dados.
    """
    z_values = np.linspace(0, z_max + 1, 1000)
    p_hit_values = [p_hit(z, z_exp, sigma) for z in z_values]
    p_short_values = [p_short(z, z_exp, lbd) for z in z_values]
    p_random_values = [p_random(z, z_max) for z in z_values]
    p_max_values = [p_max(z, z_max) for z in z_values]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(z_values, p_hit_values, color='blue')
    axs[0, 0].set_title('Hit Probability')
    axs[0, 0].set_xlabel('z values')
    axs[0, 0].set_ylabel('Probability Density')
    axs[0, 0].grid(True)

    axs[0, 1].plot(z_values, p_short_values, color='orange')
    axs[0, 1].set_title('Short Probability')
    axs[0, 1].set_xlabel('z values')
    axs[0, 1].set_ylabel('Probability Density')
    axs[0, 1].grid(True)

    axs[1, 0].plot(z_values, p_random_values, color='green')
    axs[1, 0].set_title('Random Probability')
    axs[1, 0].set_xlabel('z values')
    axs[1, 0].set_ylabel('Probability Density')
    axs[1, 0].grid(True)

    axs[1, 1].plot(z_values, p_max_values, color='red')
    axs[1, 1].set_title('Max Probability')
    axs[1, 1].set_xlabel('z values')
    axs[1, 1].set_ylabel('Probability Density')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_combined_probabilities(z_exp, z_max, params):
    """
    Plota a combinação linear das distribuições de probabilidade.
    """
    sigma = params['sigma']
    lbd = params['lambda']

    z_values = np.linspace(0, z_max + 1, 1000)
    ph_val = [p_hit(z, z_exp, sigma) for z in z_values]
    ps_val = [p_short(z, z_exp, lbd) for z in z_values]
    pm_val = [p_max(z, z_max) for z in z_values]
    pr_val = [p_random(z, z_max) for z in z_values]

    prob_total = calc_full_probabilities(ph_val, ps_val, pm_val, pr_val, params)

    plt.figure(figsize=(10, 6))
    plt.plot(z_values, prob_total, label='Combined Probability', color='purple')
    plt.title('Combined Probability Distribution')
    plt.xlabel('z values')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    z_exp = 20
    z_max = 40
    sigma = 1
    lbd = 0.05
    plot_probabilities(z_exp, z_max, sigma, lbd)
    print("Plots generated for the probability distributions.")