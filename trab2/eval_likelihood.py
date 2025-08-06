from likelihood_field import likelihood_field_range_finder_model
from plot_utils import plot_map, visualize_distance_field, viz_likelihood
from estimar_pose_hough import simulate_lidar,compute_distance_field
from world import get_world, world_to_occupancy_grid
import numpy as np

if __name__ == "__main__":
    # Load the world map
    walls_world = get_world(plot=True)
    resolution = 0.05
    grid, origin = world_to_occupancy_grid(walls_world, resolution=resolution)
    dist_field = compute_distance_field(grid, resolution=resolution)

    visualize_distance_field(dist_field, walls_world, origin,resolution=resolution)

    beam_angles = np.linspace(-np.pi/2, np.pi/2, 8)

    # Robot poses
    x_poses = np.linspace(origin[0] + 0.5, origin[0] + grid.shape[1] * resolution - 0.5, 500)
    y_poses = np.linspace(origin[1] + 0.5, origin[1] + grid.shape[0] * resolution - 0.5, 500)
    poses = np.array(np.meshgrid(x_poses, y_poses)).T.reshape(-1, 2)

    likelihood_grid = np.zeros_like(grid, dtype=float)

    z_max = 5
    z_hit = 0.6605691349400036
    z_rand = 0.2511077337010597
    sigma = 0.40716950745496716

    true_pose = (5.0, 5.0, 0.0)  # ground truth pose for testing
    z_exp_true, _ = simulate_lidar(true_pose, beam_angles, walls_world, max_range=z_max)
    z_measured = [d + np.random.normal(0, 0.5) if d is not None else z_max for d in z_exp_true]

    
    for pose in poses:
        x, y = pose
        theta = 0.0  # or any orientation you'd like to test

        likelihood = likelihood_field_range_finder_model(
            zt=z_measured,
            xt=(x, y, theta),
            distance_field=dist_field,
            origin=origin,
            resolution=resolution,
            sensor_angles=beam_angles,
            z_max=z_max,
            z_hit=z_hit,
            sigma_hit=sigma,
            z_rand=z_rand
        )

        # Store likelihood in the closest grid cell
        i = int((y - origin[1]) / resolution)
        j = int((x - origin[0]) / resolution)

        if 0 <= i < likelihood_grid.shape[0] and 0 <= j < likelihood_grid.shape[1]:
            likelihood_grid[i, j] = likelihood
    
    # Normalize for visualization
    norm_likelihood = likelihood_grid / np.max(likelihood_grid + 1e-9)

    viz_likelihood(norm_likelihood,origin,resolution,grid,walls_world)