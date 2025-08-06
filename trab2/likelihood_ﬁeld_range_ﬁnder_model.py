import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from map import load_map, make_map
from lidar_ray_casting import get_pose

# ----------------------------------------
# 1. Função para gerar grid do mapa (ocupado/livre)
# ----------------------------------------
def build_occupancy_grid(x_bounds, y_bounds, resolution, walls):
    """
    Cria um grid binário do mapa (0 = livre, 1 = obstáculo).
    """
    width = int((x_bounds[1] - x_bounds[0]) / resolution)
    height = int((y_bounds[1] - y_bounds[0]) / resolution)

    grid = np.zeros((height, width), dtype=np.uint8)

    # Preenche com obstáculos baseados nas paredes
    for (x1, y1), (x2, y2) in walls:
        num_points = int(max(abs(x2 - x1), abs(y2 - y1)) / resolution) + 1
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)

        for x, y in zip(xs, ys):
            ix = int((x - x_bounds[0]) / resolution)
            iy = int((y - y_bounds[0]) / resolution)
            if 0 <= ix < width and 0 <= iy < height:
                grid[iy, ix] = 1

    return grid

# ----------------------------------------
# 2. Construir campo de distâncias
# ----------------------------------------
def build_likelihood_field(grid, resolution):
    """
    Constrói campo de distâncias (likelihood field) usando distance transform.
    """
    # Distância até o obstáculo mais próximo (em células)
    distance_field = distance_transform_edt(1 - grid)
    # Converter para metros
    return distance_field * resolution

# ----------------------------------------
# 3. Modelo likelihood_field_range_finder
# ----------------------------------------
def likelihood_field_range_finder_model(zt, pose, beam_angles, likelihood_field,
                                        x_bounds, y_bounds, resolution, sigma=0.2):
    """
    Calcula p(z|x) usando o likelihood field.
    """
    x, y, theta = pose
    prob = 1.0
    height, width = likelihood_field.shape

    for z, angle in zip(zt, beam_angles):
        # Ponto medido no mundo
        px = x + z * np.cos(theta + angle)
        py = y + z * np.sin(theta + angle)

        # Converter para índice do grid
        ix = int((px - x_bounds[0]) / resolution)
        iy = int((py - y_bounds[0]) / resolution)

        # Verifica se está dentro do grid
        if 0 <= ix < width and 0 <= iy < height:
            d = likelihood_field[iy, ix]
        else:
            d = 1e6  # muito longe se estiver fora do mapa

        # Probabilidade gaussiana
        p = np.exp(-0.5 * (d / sigma) ** 2)
        prob *= p

    return prob

# ----------------------------------------
# 4. Simulação do heatmap
# ----------------------------------------
def simulate_likelihood_field_heatmap(beam_angles, zt, likelihood_field,
                                      x_bounds, y_bounds, resolution, sigma=0.2):
    """
    Cria um heatmap p(z|x) sobre todo o mapa usando likelihood field.
    """
    xs = np.arange(x_bounds[0], x_bounds[1], resolution)
    ys = np.arange(y_bounds[0], y_bounds[1], resolution)

    prob_grid = np.zeros((len(ys), len(xs)))

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            pose = (x, y, 0.0)  # fixamos theta=0 para simplificar
            prob_grid[iy, ix] = likelihood_field_range_finder_model(
                zt, pose, beam_angles, likelihood_field,
                x_bounds, y_bounds, resolution, sigma
            )

    # Normalização
    prob_grid /= np.sum(prob_grid)
    return xs, ys, prob_grid

# ----------------------------------------
# 5. Função principal
# ----------------------------------------
def main():
    # 1. Carregar mapa
    x_bounds, y_bounds, fig_size, walls = load_map()

    # 2. Construir grid e likelihood field
    resolution = 0.1  # tamanho da célula em metros
    occupancy_grid = build_occupancy_grid(x_bounds, y_bounds, resolution, walls)
    likelihood_field = build_likelihood_field(occupancy_grid, resolution)

    # 3. Definir feixes do LIDAR e medições simuladas
    beam_angles = np.linspace(-np.pi/2, np.pi/2, 8)  # 8 feixes
    
    # Simulação simples: distâncias ideais (poderia usar simulate_lidar)
    zt = [3.0, 4.5, 5.0, 2.5, 3.5, 4.0, 6.0, 5.5]

    # 4. Calcular mapa de probabilidade
    xs, ys, prob_grid = simulate_likelihood_field_heatmap(
        beam_angles, zt, likelihood_field, x_bounds, y_bounds, resolution
    )

    # 5. Plotar heatmap
    plt.figure(figsize=fig_size)
    plt.imshow(prob_grid, extent=[xs[0], xs[-1], ys[0], ys[-1]],
               origin='lower', cmap='hot')
    plt.colorbar(label="Probabilidade normalizada")
    plt.plot(get_pose()[0], get_pose()[1], 'bo', label='Pose real')
    plt.legend()
    plt.title("Mapa de Probabilidade - Likelihood Field Range Finder Model")
    plt.show()

if __name__ == "__main__":
    main()
