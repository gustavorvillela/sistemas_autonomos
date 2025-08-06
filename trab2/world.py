import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from plot_utils import plot_map
from map import load_map, make_map


def pixel_to_world(lines_px, img_shape):
    """
    Converte linhas detectadas em pixels para coordenadas do mundo.
    world_bounds = (x_min, x_max, y_min, y_max)
    """
    x_bounds,y_bounds,_,_ = load_map()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    height, width = img_shape[:2]

    walls_world = []
    for (x1, y1), (x2, y2) in lines_px:
        wx1 = x_min + (x1 / width)  * (x_max - x_min)
        wy1 = y_min + (1 - y1 / height) * (y_max - y_min)  # invertendo eixo Y
        wx2 = x_min + (x2 / width)  * (x_max - x_min)
        wy2 = y_min + (1 - y2 / height) * (y_max - y_min)
        walls_world.append(((wx1, wy1), (wx2, wy2)))

    return walls_world

def renders_to_array(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    rho = 1                 # pixel resolution
    theta = np.pi / 180     # angular resolution
    threshold = 10          # lower = more sensitive
    min_line_length = 10    # smaller walls get detected
    max_line_gap = 10       # join nearby line fragments

    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    walls_detected = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            walls_detected.append(((x1, y1), (x2, y2)))
    return walls_detected
  
def get_world(plot=True):
    # Cria figura
    fig = make_map()

    # Renderiza para array
    image = renders_to_array(fig)
    if plot:
        plot_map(image)

    # Converte e detecta bordas
    walls_detected = hough(image)


    walls_world = pixel_to_world(walls_detected, image.shape)
    return walls_world

def world_to_occupancy_grid(walls, resolution=0.05, padding=1.0):
    """
    Converts wall segments to an occupancy grid.
    
    Args:
        walls: list of ((x1, y1), (x2, y2)) line segments (in meters)
        resolution: meters per cell
        padding: extra margin (in meters) around all walls

    Returns:
        occupancy_map: 2D binary numpy array (1 = wall/occupied, 0 = free)
        origin: (x_offset, y_offset) so you can map world -> grid
    """
    # Compute bounding box
    all_points = np.array([pt for wall in walls for pt in wall])
    min_x, min_y = np.min(all_points, axis=0) - padding
    max_x, max_y = np.max(all_points, axis=0) + padding

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # Create blank image (rows = height, cols = width)
    grid = np.zeros((height, width), dtype=np.uint8)

    # Rasterize each wall segment onto the grid
    for (x1, y1), (x2, y2) in walls:
        p1 = ((x1 - min_x) / resolution, (y1 - min_y) / resolution)
        p2 = ((x2 - min_x) / resolution, (y2 - min_y) / resolution)
        cv2.line(grid,
                 (int(p1[0]), int(height - p1[1])),  # (x, y) → row-col image format
                 (int(p2[0]), int(height - p2[1])),
                 color=1, thickness=1)

    return grid, (min_x, min_y)
