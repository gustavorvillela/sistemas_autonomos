import numpy as np
import math

from world import get_world

def ray_intersection(pose, angle, walls_world):
    """
    Calcula a interseção de um raio com as paredes do mapa
    Retorna a distância até a parede mais próxima na direção do raio
    """
    x, y, theta = pose
    ray_dir = theta + angle
    
    # Parametrização do raio: (x + t*cos(ray_dir), y + t*sin(ray_dir))
    closest_distance = math.inf
    intersection_point = None
    
    for wall in walls_world:
        (x1, y1), (x2, y2) = wall
        
        # Parametrização do segmento de parede: (x1 + s*(x2-x1), y1 + s*(y2-y1))
        # Resolver sistema para encontrar t e s
        dx_wall = x2 - x1
        dy_wall = y2 - y1
        dx_ray = math.cos(ray_dir)
        dy_ray = math.sin(ray_dir)
        
        denominator = dy_wall * dx_ray - dx_wall * dy_ray
        
        if abs(denominator) < 1e-6:  # Linhas paralelas
            continue
            
        s = (dx_ray * (y - y1) - dy_ray * (x - x1)) / denominator
        t = (dx_wall * (y - y1) - dy_wall * (x - x1)) / denominator
        
        if 0 <= s <= 1 and t > 0:  # Interseção dentro do segmento e na direção do raio
            if t < closest_distance:
                closest_distance = t
                intersection_point = (x + t*dx_ray, y + t*dy_ray)
    
    return closest_distance if closest_distance != math.inf else None, intersection_point

def simulate_lidar(pose, beam_angles, walls_world, max_range=20.0):
    """
    Simula leituras de um sensor lidar
    Retorna distâncias e pontos de interseção para cada ângulo
    """
    distances = []
    points = []
    
    for angle in beam_angles:
        dist, point = ray_intersection(pose, angle, walls_world)
        if dist is not None and dist <= max_range:
            distances.append(dist)
            points.append(point)
        else:
            distances.append(max_range)
            points.append(None)
    
    return distances, points

def estimate_position(measurements, walls_world, initial_guess=(5.0, 5.0,math.pi/4), max_iter=100):
    """
    Estimativa simples de posição usando mínimos quadrados
    (Implementação básica - pode ser melhorada)
    """
    from scipy.optimize import minimize
    
    def cost_function(params):
        x, y, theta = params
        pose = (x, y, theta)
        simulated_dists, _ = simulate_lidar(pose, beam_angles, walls_world)
        
        # Calcular erro entre medições simuladas e reais
        error = 0
        for sim_dist, meas_dist in zip(simulated_dists, measurements):
            if meas_dist is not None and sim_dist is not None:
                error += (sim_dist - meas_dist)**2
        return error
    
    # Otimização para encontrar a pose que minimiza o erro
    result = minimize(cost_function, initial_guess, method='L-BFGS-B')
    return result.x


walls_world = get_world()

# robo
pose = (5.0, 5.0, np.pi/4)

# angulos de raio
beam_angles = np.linspace(-np.pi/2, np.pi/2, 8)

true_distances, true_points = simulate_lidar(pose, beam_angles, walls_world)

noisy_distances = [d + np.random.normal(0, 0.1) if d is not None else None 
                    for d in true_distances]

estimated_pose = estimate_position(noisy_distances, walls_world)

print(f"Pose real: {pose}")
print(f"Pose estimada: {estimated_pose}")