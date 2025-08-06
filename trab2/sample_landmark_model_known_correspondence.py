import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from landmark_model_known_correspondence import (
    load_landmarks,
    plot_landmarks,
    plot_robot,
    simulate_landmark_detections
)
from map import load_map

def sample_landmark_model_known_correspondence(detection, landmarks, num_samples=100, sigma_r=0.1, sigma_phi=0.05):
    lid, r_meas, phi_meas = detection
    landmark_dict = {lid_: (lx, ly) for (lid_, lx, ly) in landmarks}
    lx, ly = landmark_dict[lid]
    samples = []
    for _ in range(num_samples):
        gamma = np.random.uniform(0, 2 * np.pi)
        r_hat = r_meas + np.random.normal(0, sigma_r)
        phi_hat = phi_meas + np.random.normal(0, sigma_phi)
        x = lx + r_hat * np.cos(gamma)
        y = ly + r_hat * np.sin(gamma)
        theta = gamma - np.pi - phi_hat
        samples.append((x, y, theta))
    return samples

# ---------------------------
# Animação
# ---------------------------
def main():
    x_bounds, y_bounds, fig_size, _ = load_map()
    landmarks = load_landmarks("config_landmarks.txt")

    # Movimento do robô
    x_s = np.linspace(5, 7, 40) 
    y_s = np.linspace(5, 11, 40)

    colors = ['green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    fig, ax = plt.subplots(figsize=fig_size)

    def update(frame):
        ax.clear()
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
        ax.set_aspect('equal')

        # Pose atual
        x = x_s[frame]
        y = y_s[frame]
        pose_real = (x, y, np.pi/4 + frame/10)

        # Plot mapa e robô
        plot_landmarks(ax, landmarks)
        plot_robot(ax, pose_real)

        # Simular deteções
        detections = simulate_landmark_detections(pose_real, landmarks)
        for idx, detection in enumerate(detections):
            color = colors[idx % len(colors)]
            samples = sample_landmark_model_known_correspondence(detection, landmarks, num_samples=300)
            xs = [s[0] for s in samples]
            ys = [s[1] for s in samples]
            ax.scatter(xs, ys, s=10, color=color, alpha=0.5, label=f"Landmark {detection[0]}")

        # Legenda sem duplicatas
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        ax.set_title("Sample Landmark Model - Vários Landmarks")

    # Criar animação
    ani = FuncAnimation(fig, update, frames=len(x_s), interval=100)

    # Salvar como GIF
    ani.save("sample_landmark.gif", writer=PillowWriter(fps=10))

if __name__ == "__main__":
    main()

