import matplotlib.pyplot as plt
import ast

def load_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Usa ast.literal_eval para converter strings em tuplas/listas
                try:
                    config[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError) as e:
                    print(f"Erro ao analisar a linha: {line}. Erro: {e}")
    return config

def load_map():
  config = load_config('config_map.txt')
  x_bounds = config['x_bounds']
  y_bounds = config['y_bounds']
  fig_size = config['fig_size']
  walls = config['walls']
  return x_bounds,y_bounds,fig_size,walls

def make_map():
    # Carrega o arquivo
    x_bounds,y_bounds,fig_size,walls = load_map()
    #world_bounds = x_bounds + y_bounds
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect('equal')
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2)
    plt.axis('off')
    plt.close()
    return fig

