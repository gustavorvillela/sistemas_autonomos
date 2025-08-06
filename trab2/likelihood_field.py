import numpy as np
import sys
sys.path.append('../trab1')
from calculate_prob import prob_normal_distribution

def likelihood_field_range_finder_model(zt, xt, distance_field, origin, resolution,
                                 sensor_angles, xk_sens=0.0, yk_sens=0.0,
                                 z_max=10.0, z_hit=0.8, sigma_hit=0.2, z_rand=0.2):
    q = 1.0
    x, y, theta = xt
    ox, oy = origin
    height, width = distance_field.shape

    for k, z in enumerate(zt):
        if z == z_max or np.isnan(z) or np.isinf(z):
            continue

        theta_k = sensor_angles[k]

        x_z = x + xk_sens * np.cos(theta) - yk_sens * np.sin(theta) + z * np.cos(theta + theta_k)
        y_z = y + yk_sens * np.cos(theta) + xk_sens * np.sin(theta) + z * np.sin(theta + theta_k)

        map_x = int((x_z - ox) / resolution)
        map_y = int((y_z - oy) / resolution)
        map_y = height - map_y  # invert Y axis to match image

        if 0 <= map_x < width and 0 <= map_y < height:
            dist = distance_field[map_y, map_x]
        else:
            dist = z_max

        p = z_hit * prob_normal_distribution(dist, sigma_hit**2) + z_rand / z_max
        q *= p

    return q