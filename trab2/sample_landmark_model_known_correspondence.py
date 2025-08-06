import numpy as np
import matplotlib.pyplot as plt
from landmark_model_known_correspondence import (
    load_landmarks,
    plot_landmarks,
    plot_robot,
    simulate_landmark_detections
)
from map import load_map

# ---------------------------
# 1. Função para amostrar poses do robô
# ---------------------------
def sample_landmark_model_known_correspondence(detection, landmarks, num_samples=100, sigma_r=0.1, sigma_phi=0.05):
    """
    Gera amostras de possíveis poses do robô com base na medição de um landmark conhecido.

    detection: tupla (id, r_medido, phi_medido)
    landmarks: lista [(id, x, y)]
    num_samples: número de amostras a gerar
    sigma_r, sigma_phi: desvio padrão do ruído de distância e ângulo

    Retorna: lista de poses [(x,y,theta), ...]
    """
    lid, r_meas, phi_meas = detection

    # Achar coordenadas do landmark
    landmark_dict = {lid_: (lx, ly) for (lid_, lx, ly) in landmarks}
    if lid not in landmark_dict:
        raise ValueError(f"Landmark ID {lid} não encontrado")

    lx, ly = landmark_dict[lid]

    samples = []
    for _ in range(num_samples):
        # Passo 1: sortear ângulo aleatório uniforme
        gamma = np.random.uniform(0, 2 * np.pi)

        # Passo 2: adicionar ruído na medição
        r_hat = r_meas + np.random.normal(0, sigma_r)
        phi_hat = phi_meas + np.random.normal(0, sigma_phi)

        # Passo 3: calcular posição do robô a partir do landmark
        x = lx + r_hat * np.cos(gamma)
        y = ly + r_hat * np.sin(gamma)

        # Passo 4: calcular orientação
        theta = gamma - np.pi - phi_hat

        samples.append((x, y, theta))

    return samples

# ---------------------------
# 2. Função principal de simulação
# ---------------------------
def main():
    # 1. Carregar mapa e landmarks
    x_bounds, y_bounds, fig_size, _ = load_map()
    landmarks = load_landmarks("config_landmarks.txt")

    # 2. Pose real do robô
    pose_real = (5, 5, np.pi/4)

    # 3. Simular detecção de landmarks
    detections = simulate_landmark_detections(pose_real, landmarks)

    if len(detections) == 0:
        print("Nenhum landmark detectado.")
        return

    # Para simplicidade, usar o primeiro detection
    detection = detections[0]

    # 4. Gerar amostras de pose
    samples = sample_landmark_model_known_correspondence(detection, landmarks, num_samples=200)

    # 5. Plotar resultado
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')

    # Plotar landmarks e robô real
    plot_landmarks(ax, landmarks)
    plot_robot(ax, pose_real)

    # Plotar amostras como pontos verdes
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    ax.scatter(xs, ys, s=10, color='green', alpha=0.5, label='Amostras')

    ax.set_title("Sample Landmark Model (Known Correspondence)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
