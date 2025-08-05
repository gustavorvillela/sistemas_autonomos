import matplotlib.pyplot as plt
import numpy as np
import ast
from map import load_map
from matplotlib.axes import Axes
# ---------------------------
# 1. Carregar marcos do arquivo
# ---------------------------
def load_landmarks(file_path="config_landmarks.txt"):
    """
    Carrega posições (id,x,y) dos marcos a partir de um arquivo de configuração.
    Formato do arquivo:
        landmarks = [(1,2,2), (2,5,5), (3,8,3)]
    """
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                try:
                    config[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError) as e:
                    print(f"Erro ao analisar linha: {line}. Erro: {e}")

    if "landmarks" not in config:
        raise ValueError("Arquivo de configuração não contém 'landmarks'")
    return config["landmarks"]

# ---------------------------
# 2. Função para plotar robô
# ---------------------------
def plot_robot(ax:Axes, pose, robot_radius=0.3):
    """
    Plota o robô como um círculo com seta de orientação.
    pose: (x, y, theta)
    """
    x, y, theta = pose
    # Círculo representando o robô
    robot = plt.Circle((x, y), robot_radius, color='blue', alpha=0.5)
    ax.add_patch(robot)

    # Seta indicando orientação
    ax.arrow(
        x, y,
        0.5 * np.cos(theta),  # deslocamento x da seta
        0.5 * np.sin(theta),  # deslocamento y da seta
        head_width=0.1,
        color='blue'
    )

# ---------------------------
# 3. Função para plotar marcos
# ---------------------------
def plot_landmarks(ax:Axes, landmarks):
    """
    Plota marcos (landmarks) como pontos vermelhos no mapa.
    landmarks: lista de tuplas [(x,y), ...]
    """
    for (id,lx, ly) in landmarks:
      ax.plot(lx, ly, 'ro', markersize=8, label='Landmark')
      ax.text(
            lx + 0.2,  # Deslocamento em X para evitar sobreposição
            ly + 0.2,  # Deslocamento em Y para evitar sobreposição
            f"ID: {id}",  # Texto exibido (ID do landmark)
            fontsize=10,  # Tamanho da fonte
            color='black',  # Cor do texto
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # Fundo branco para melhor legibilidade
        )
# ---------------------------
# 5. Função para simular detecção de landmarks
# ---------------------------
def simulate_landmark_detections(pose, landmarks, max_range=10.0, sigma_r=0.1, sigma_phi=0.05):
    """
    Simula a detecção de landmarks conhecidos a partir da pose do robô.
    Retorna lista de medições: [(id, distancia, angulo_relativo), ...]
    
    pose: (x, y, theta)
    landmarks: lista [(id, x, y)]
    max_range: distância máxima de detecção
    sigma_r: desvio padrão do ruído de distância (m)
    sigma_phi: desvio padrão do ruído de ângulo (rad)
    """
    x_r, y_r, theta_r = pose
    detections = []

    for (land_id, lx, ly) in landmarks:
        # 1. Distância e ângulo reais
        dx = lx - x_r
        dy = ly - y_r
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) - theta_r

        # Normaliza ângulo para [-pi, pi]
        angle = (angle + np.pi) % (2*np.pi) - np.pi

        # 2. Verifica se está dentro do alcance
        if dist <= max_range:
            # 3. Adiciona ruído
            noisy_dist = dist + np.random.normal(0, sigma_r)
            noisy_angle = angle + np.random.normal(0, sigma_phi)

            detections.append((land_id, noisy_dist, noisy_angle))

    return detections

def landmark_model_known_correspondence(pose, detections, landmarks, sigma_r=0.1, sigma_phi=0.05):
    """
    Calcula probabilidade p(z | x) dado pose e landmarks conhecidos com correspondência.
    pose: (x, y, theta)
    detections: [(id, r_medido, phi_medido), ...]
    landmarks: [(id, x_l, y_l), ...]
    """
    x_r, y_r, theta_r = pose
    prob = 1.0

    # Dicionário para acesso rápido aos landmarks por ID
    landmark_dict = {lid: (lx, ly) for (lid, lx, ly) in landmarks}

    for (lid, r_meas, phi_meas) in detections:
        # Coordenadas do landmark correspondente
        if lid not in landmark_dict:
            continue
        lx, ly = landmark_dict[lid]

        # Distância e ângulo esperados
        dx = lx - x_r
        dy = ly - y_r
        r_exp = np.sqrt(dx**2 + dy**2)
        phi_exp = np.arctan2(dy, dx) - theta_r
        phi_exp = (phi_exp + np.pi) % (2*np.pi) - np.pi  # normaliza [-pi,pi]

        # Probabilidades individuais
        p_r = (1 / (np.sqrt(2*np.pi) * sigma_r)) * np.exp(-0.5 * ((r_meas - r_exp) / sigma_r)**2)
        p_phi = (1 / (np.sqrt(2*np.pi) * sigma_phi)) * np.exp(-0.5 * ((phi_meas - phi_exp) / sigma_phi)**2)

        prob *= p_r * p_phi

    return prob
def simulate_landmark_model_map(detections, landmarks, x_bounds, y_bounds, grid_res=1.0, sigma_r=0.5, sigma_phi=0.2):
    """
    Cria um mapa de calor probabilístico p(z|x) para cada posição (x,y) usando landmark_model_known_correspondence.
    """
    xs = np.arange(x_bounds[0], x_bounds[1] + grid_res, grid_res)
    ys = np.arange(y_bounds[0], y_bounds[1] + grid_res, grid_res)

    prob_grid = np.zeros((len(ys), len(xs)))

    # Varre cada posição do grid (fixando theta=0 para simplificar ou pode variar também)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            pose = (x, y, 0.0)  # aqui theta fixo = 0
            prob_grid[iy, ix] = landmark_model_known_correspondence(
                pose, detections, landmarks, sigma_r, sigma_phi
            )

    # Normaliza
    prob_grid /= np.sum(prob_grid)
    return xs, ys, prob_grid

def main():
    # Carregar mapa (paredes)
    x_bounds, y_bounds, fig_size, _ = load_map()

    # Criar figura e eixos
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')

    # Carregar e plotar marcos
    landmarks = load_landmarks("config_landmarks.txt")
    plot_landmarks(ax, landmarks)

    # Definir pose do robô
    pose = (5, 5, np.pi / 4)  # Exemplo: posição (2,2) orientação 45 graus
    plot_robot(ax, pose)
    
    # Título e exibição
    ax.set_title("Mapa com Marcos e Robô")

    
    detections = simulate_landmark_detections(pose, landmarks)
    xs, ys, prob_grid = simulate_landmark_model_map(detections, landmarks, x_bounds, y_bounds, grid_res=0.2)
    plt.figure(figsize=fig_size)
    plt.imshow(prob_grid, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower', cmap='hot')
    plt.colorbar(label="Probabilidade normalizada")
    plt.plot(pose[0], pose[1], 'bo', label='Pose real')
    plt.legend()
    plt.title("Mapa de Probabilidade - Landmark Model (Known Correspondence)")
    plt.show()
if __name__ == "__main__":
    main()
