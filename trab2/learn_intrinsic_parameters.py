import numpy as np
import math
from beam_range_finder_model import p_hit, p_short, p_max, p_rand, simulate_beam_model_map,get_pose
from lidar_ray_casting import simulate_lidar,get_beam_angles
from world import get_world
from numpy.typing import NDArray
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA
import random

def learn_intrinsic_parameters(Z:list[list[float]], X: list[tuple[float,float,float]], walls_world: list,beam_angles:NDArray, max_range:float=20.0, 
                                init_params: dict[str,float]|None=None, max_iter:int=50, tol:float=1e-3) -> dict[str,float]:
    """
    Aprende automaticamente os parâmetros intrínsecos do modelo beam_range_finder
    usando Expectation-Maximization (EM).

    Parâmetros:
    -----------
    Z : lista de listas
        Conjunto de leituras do LIDAR (cada elemento é uma lista de distâncias para um pose).
    X : lista de tuplas
        Conjunto de poses correspondentes (x, y, theta) para cada leitura em Z.
    walls_world : lista
        Paredes do mapa (coordenadas do mundo).
    beam_angles: NDArray
        Array de ângulos correspondentes ao ray casting.
    max_range : float
        Alcance máximo do sensor.
    init_params : dict ou None
        Parâmetros iniciais (z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short).
        Se None, valores padrão serão usados.
    max_iter : int
        Número máximo de iterações do EM.
    tol : float
        Critério de convergência (diferença mínima entre iterações).

    Retorna:
    --------
    params : dict
        Parâmetros ajustados (z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short)
    """

    # -------------------------
    # 1. Inicialização
    # -------------------------
    if init_params is None:
        params = {
            "z_hit": 0.7,
            "z_short": 0.1,
            "z_max": 0.1,
            "z_rand": 0.1,
            "sigma_hit": 0.2,
            "lambda_short": 1.0
        }
    else:
        params = init_params.copy()

    # Normaliza pesos (z_hit + z_short + z_max + z_rand = 1)
    total_weight = params["z_hit"] + params["z_short"] + params["z_max"] + params["z_rand"]
    for key in ["z_hit", "z_short", "z_max", "z_rand"]:
        params[key] /= total_weight

    # -------------------------
    # 2. Iteração EM
    # -------------------------
    for iteration in range(max_iter):
        # Acumuladores para atualização
        sum_e_hit = 0.0
        sum_e_short = 0.0
        sum_e_max = 0.0
        sum_e_rand = 0.0
        weighted_error_hit = 0.0
        weighted_count_hit = 0.0
        weighted_sum_short = 0.0
        weighted_z_short = 0.0

        # Para cada leitura e pose
        for zt, xt in zip(Z, X):
            # Distâncias esperadas via ray casting
            z_expected, _ = simulate_lidar(xt,beam_angles , walls_world, max_range=max_range)

            for z, z_star in zip(zt, z_expected):
                # -------------------------
                # E-step: calcular responsabilidades
                # -------------------------
                p1 = params["z_hit"]   * p_hit(z, z_star, params["sigma_hit"], max_range)
                p2 = params["z_short"] * p_short(z, z_star, params["lambda_short"])
                p3 = params["z_max"]   * p_max(z, max_range)
                p4 = params["z_rand"]  * p_rand(z, max_range)

                eta = p1 + p2 + p3 + p4  # normalizador
                if eta == 0:
                    continue  # ignora feixe inválido

                e_hit = p1 / eta
                e_short = p2 / eta
                e_max = p3 / eta
                e_rand = p4 / eta

                # -------------------------
                # Acumular para M-step
                # -------------------------
                sum_e_hit += e_hit
                sum_e_short += e_short
                sum_e_max += e_max
                sum_e_rand += e_rand

                weighted_error_hit += e_hit * ((z - z_star) ** 2)
                weighted_count_hit += e_hit

                weighted_sum_short += e_short
                weighted_z_short += e_short * z

        # -------------------------
        # M-step: atualizar parâmetros
        # -------------------------
        N = sum_e_hit + sum_e_short + sum_e_max + sum_e_rand

        new_params = {
            "z_hit":   sum_e_hit / N,
            "z_short": sum_e_short / N,
            "z_max":   sum_e_max / N,
            "z_rand":  sum_e_rand / N,
            "sigma_hit": math.sqrt(weighted_error_hit / (weighted_count_hit + 1e-9)),
            "lambda_short": weighted_sum_short / (weighted_z_short + 1e-9)
        }

        # Critério de convergência: diferença entre parâmetros antigos e novos
        diff = sum(abs(params[k] - new_params[k]) for k in params)
        params = new_params

        if diff < tol:
            print(f"Convergiu em {iteration+1} iterações.")
            break

    return params

def calculate_dispersion(prob_grid):
    """
    Calcula dispersão (entropia) do grid de probabilidade.
    Quanto menor a entropia, mais concentrada a distribuição.
    """
    # Evita log(0)
    prob_flat = prob_grid.flatten()
    prob_flat = prob_flat[prob_flat > 0]  # remove zeros
    entropy = -np.sum(prob_flat * np.log(prob_flat))
    return entropy

def generate_pose_sets(n: int, L: float, t: float) -> list[tuple[float,float,float]]:
    """
    Gera uma lista de n pose_sets, cada um contendo 3 poses aleatórias.
    
    Args:
        n: Número de pose_sets a serem gerados
        L: Valor máximo para cada elemento nas poses (x, y)
        t: Valor máximo para theta
        
    Returns:
        Lista de pose_sets no formato:
            [(x1,y1,θ1), (x2,y2,θ2), (x3,y3,θ3), ....]
    """
    pose_sets = []
    
    for _ in range(n):
        x = random.uniform(0, L)
        y = random.uniform(0, L)
        theta = random.uniform(0, t)
        pose_sets.append((x, y, theta))
    
    return pose_sets

def generate_random_params(n=20):
    """
    Gera n conjuntos de parâmetros aleatórios plausíveis.
    Intervalos baseados no modelo probabilístico típico.
    """
    params_list = []
    for _ in range(n):
        # Pesos aleatórios para z_hit, z_short, z_max, z_rand que somem 1
        weights = np.random.dirichlet(np.ones(4))  # soma = 1
        params_list.append([
            weights[0],  # z_hit
            weights[1],  # z_short
            weights[2],  # z_max
            weights[3],  # z_rand
            np.random.uniform(0.1, 1.0),  # sigma_hit
            np.random.uniform(0.5, 2.0),  # lambda_short
            20.0  # max_range fixo
        ])
    return params_list

def main():
    # 1. Carregar mapa
    walls_world = get_world()

    # 2. Aprender parâmetros ótimos usando dataset real/simulado
    pose_sets = generate_pose_sets(15, 10, 3)  # dataset para treinamento
    Z = []
    for pose in pose_sets:
        dists, _ = simulate_lidar(pose, get_beam_angles(), walls_world)
        Z.append([d + np.random.normal(0, 0.1) for d in dists])

    best_params = learn_intrinsic_parameters(Z, pose_sets, walls_world, get_beam_angles())
    best_params["max_range"] = 20.0

    # Converter para vetor
    best_vector = [
        best_params["z_hit"],
        best_params["z_short"],
        best_params["z_max"],
        best_params["z_rand"],
        best_params["sigma_hit"],
        best_params["lambda_short"],
        best_params["max_range"]
    ]

    # Calcular dispersão do melhor
    true_dists, _ = simulate_lidar(get_pose(), get_beam_angles(), walls_world)
    noisy_measurements = [d + np.random.normal(0, 0.1) for d in true_dists]
    xs, ys, prob_grid = simulate_beam_model_map(walls_world, noisy_measurements, best_params, get_beam_angles())
    best_dispersion = calculate_dispersion(prob_grid)

    # 3. Gerar parâmetros aleatórios para comparação
    random_params_list = generate_random_params(30)

    all_vectors = [best_vector] + random_params_list
    dispersions = [best_dispersion]

    # Calcular dispersão para cada parâmetro aleatório
    for vec in random_params_list:
        # Reconstruir dict para passar no simulate_beam_model_map
        params_dict = {
            "z_hit": vec[0],
            "z_short": vec[1],
            "z_max": vec[2],
            "z_rand": vec[3],
            "sigma_hit": vec[4],
            "lambda_short": vec[5],
            "max_range": 20.0
        }

        xs, ys, prob_grid = simulate_beam_model_map(walls_world, noisy_measurements, params_dict, get_beam_angles())
        dispersions.append(calculate_dispersion(prob_grid))

    # 4. PCA (7D → 1D)
    all_vectors = np.array(all_vectors)
    pca = PCA(n_components=1)
    params_1d = pca.fit_transform(all_vectors).flatten()

    # 5. Identificar menor dispersão (aleatório)
    best_random_idx = 1 + np.argmin(dispersions[1:])  # +1 pq 0 é o parâmetro aprendido

    # 6. Mensagem no terminal
    if best_random_idx == 0 or dispersions[0] <= dispersions[best_random_idx]:
        print("Parâmetro do learn_intrinsic_parameters é o de menor dispersão (ou empatado)!")
        print(f"dispersão do learn_intrinsic_parameters: {best_dispersion}")
        print(f"dispersão do menor parametro: {min(dispersions)}")
        print(f"dispersão do segundo menor parametro: {sorted(dispersions)[1]}")
    else:
        print("Parâmetro do learn_intrinsic_parameters NÃO é o de menor dispersão.")
        print(f"Entropia do aprendido: {dispersions[0]:.4f}")
        print(f"Menor entropia aleatória: {dispersions[best_random_idx]:.4f} (índice {best_random_idx})")

    # 7. Plotar gráfico
    plt.figure(figsize=(8,6))
    plt.scatter(params_1d[1:], dispersions[1:], color='blue', label='Parâmetros Aleatórios')

    # Ponto do parâmetro aprendido
    plt.scatter(params_1d[0], dispersions[0], color='red', s=120, label='Learn Intrinsic Parameters')

    # Ponto do menor aleatório
    plt.scatter(params_1d[best_random_idx], dispersions[best_random_idx],
                color='green', s=120, label='Menor dispersão aleatória')

    plt.title("PCA (Parâmetros) x Dispersão da Probabilidade")
    plt.xlabel("PCA 1D")
    plt.ylabel("Entropia (Dispersão)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
