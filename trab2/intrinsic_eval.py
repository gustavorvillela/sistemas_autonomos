import numpy as np
import matplotlib.pyplot as plt

from intrinsic_param import learn_intrinsic_parameters
from plot_utils import plot_combined_probabilities
from probabilities import p_hit, p_short, p_random, p_max, calc_full_probabilities

from world import get_world
from estimar_pose_hough import simulate_lidar



if __name__ == "__main__":

    # Dados de exemplo
    z_max = 5
    N = 500

    # Carregar o mundo
    walls_world = get_world()

    # Pose(s) do robô
    poses_num = 500
    poses_x = np.random.uniform(1,9,poses_num)
    poses_y = np.random.uniform(1,9,poses_num)

    # Ângulos de raio
    beam_angles = np.random.uniform(-np.pi/2, np.pi/2, poses_num)
    poses = [(x, y, beam_angles[i]) for i, (x, y) in enumerate(zip(poses_x, poses_y))]

    Z = []
    Z_exp = []
    for pose in poses:
        z_exp_i, _ = simulate_lidar(pose, beam_angles, walls_world)
        z_measured = [d + np.random.normal(0, 0.5) if d is not None else 0.0 for d in z_exp_i]
        
        Z.extend(z_measured)
        Z_exp.extend(z_exp_i)

    Z = np.array(Z)
    Z_exp = np.array(Z_exp)

    # Parâmetros iniciais
    z_init = (0.7, 0.1, 0.1, 0.1, 0.2, 0.01)
    # Aprender parâmetros intrínsecos
    params = learn_intrinsic_parameters(Z, Z_exp, z_max, num_iters=N, params_init=z_init)

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

    



