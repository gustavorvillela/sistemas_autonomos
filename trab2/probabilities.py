import numpy as np
import math

def p_hit(z, z_exp, sigma):
    """Calcula a probabilidade de um ponto z dado a média mu e desvio padrão sigma."""
    coeff = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((z - z_exp) / sigma) ** 2
    return coeff * math.exp(exponent)


def p_short(z, z_exp, lbd):
    """Calcula a probabilidade de um ponto z dado o comprimento máximo lambda."""
    if z < 0 or z > z_exp:
        return 0
    return lbd * math.exp(-lbd * z)

def p_random(z, z_max):
    """Calcula a probabilidade de um ponto z aleatório."""
    if z < 0:
        return 0
    return 1 / z_max

def p_max(z, z_max):
    """Calcula a probabilidade de um ponto z máximo."""
    return 1 if abs(z - z_max) <= 0.1 else 0


def calc_full_probabilities(ph_val, ps_val, pm_val, pr_val, theta):

    z_params = np.array([theta['z_hit'], theta['z_short'], theta['z_max'], theta['z_rand']])
    prob_vals = np.array([ph_val, ps_val, pm_val, pr_val])

    prob_total = np.dot(z_params, prob_vals)

    return prob_total