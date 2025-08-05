import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

from map import load_map, make_map

def renders_to_array(fig):
    """
    Converte uma figura matplotlib em um array numpy (imagem)
    """
    # Cria um canvas (tela de desenho) para a figura
    canvas = FigureCanvasAgg(fig)
    # Renderiza a figura no canvas
    canvas.draw()
    # Obtém o buffer de pixels (RGBA)
    buf = canvas.buffer_rgba()
    # Converte para array numpy
    image = np.asarray(buf)
    # Converte de RGBA (4 canais) para RGB (3 canais)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def hough(image):
    """
    Detecta linhas em uma imagem usando a transformada de Hough
    """
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Detecta bordas usando o algoritmo Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Aplica transformada de Hough para detectar linhas
    lines = cv2.HoughLinesP(
        edges,              # Imagem de bordas
        1,                  # Resolução de rho (pixels)
        np.pi/180,          # Resolução de theta (radianos)
        150,                # Limiar de votos mínimo
        minLineLength=20,   # Comprimento mínimo da linha
        maxLineGap=5        # Máxima lacuna permitida entre segmentos
    )
    
    walls_detected = []
    if lines is not None:
        # Para cada linha detectada
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adiciona as coordenadas da linha à lista
            walls_detected.append(((x1, y1), (x2, y2)))
    return walls_detected

def pixel_to_world(lines_px, img_shape):
    """
    Converte linhas detectadas em pixels para coordenadas do mundo real.
    Parâmetros:
        lines_px - linhas detectadas em coordenadas de pixel
        img_shape - dimensões da imagem (altura, largura)
    
    Retorna:
        Lista de paredes em coordenadas do mundo ((x1,y1), (x2,y2))
    """
    # Carrega os limites do mapa (x_min, x_max, y_min, y_max)
    x_bounds, y_bounds, _, _ = load_map()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    # Obtém altura e largura da imagem
    height, width = img_shape[:2]

    walls_world = []
    # Para cada linha detectada na imagem (em pixels)
    for (x1, y1), (x2, y2) in lines_px:
        # Converte coordenada X de pixel para mundo
        wx1 = x_min + (x1 / width) * (x_max - x_min)
        # Converte coordenada Y (invertendo o eixo Y pois imagens têm Y para baixo)
        wy1 = y_min + (1 - y1 / height) * (y_max - y_min)
        wx2 = x_min + (x2 / width) * (x_max - x_min)
        wy2 = y_min + (1 - y2 / height) * (y_max - y_min)
        
        # Adiciona a parede convertida à lista
        walls_world.append(((wx1, wy1), (wx2, wy2)))

    return walls_world

def get_world():
    """
    Função principal que obtém as paredes do mundo real
    """
    # 1. Cria a figura do mapa usando a função do módulo map
    fig = make_map()

    # 2. Converte a figura em uma imagem (array numpy)
    image = renders_to_array(fig)

    # 3. Detecta linhas/arestas na imagem
    walls_detected = hough(image)

    # 4. Converte as linhas de pixels para coordenadas do mundo
    walls_world = pixel_to_world(walls_detected, image.shape)
    
    return walls_world