import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, minLineLength=20, maxLineGap=5)
    walls_detected = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            walls_detected.append(((x1, y1), (x2, y2)))
    return walls_detected
  
def get_world():
  # Cria figura
  fig = make_map()

  # Renderiza para array
  image = renders_to_array(fig)

  # Converte e detecta bordas
  walls_detected = hough(image)


  walls_world = pixel_to_world(walls_detected, image.shape)
  return walls_world

  
