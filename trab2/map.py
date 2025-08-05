import matplotlib.pyplot as plt
import ast

# Função para carregar um arquivo de configuração
def load_config(file_path):
    # Cria um dicionário vazio para armazenar as configurações
    config = {}
    
    # Abre o arquivo no modo leitura ('r' = read)
    with open(file_path, 'r') as file:
        # Lê o arquivo linha por linha
        for line in file:
            # Verifica se a linha contém um sinal de igual (é uma configuração)
            if '=' in line:
                # Divide a linha em duas partes: antes e depois do '='
                key, value = line.split('=', 1)
                # Remove espaços em branco do início e fim da chave
                key = key.strip()
                # Remove espaços em branco do início e fim do valor
                value = value.strip()
                
                # Tenta converter o valor (que é texto) em uma estrutura de dados Python
                try:
                    config[key] = ast.literal_eval(value)
                # Se houver erro na conversão, mostra uma mensagem mas continua executando
                except (ValueError, SyntaxError) as e:
                    print(f"Erro ao analisar a linha: {line}. Erro: {e}")
    
    # Retorna o dicionário com todas as configurações
    return config

# Função específica para carregar configurações do mapa
def load_map():
    # Carrega o arquivo de configuração 'config_map.txt'
    config = load_config('config_map.txt')
    
    # Extrai os limites do eixo X (esquerda/direita)
    x_bounds = config['x_bounds']
    # Extrai os limites do eixo Y (baixo/cima)
    y_bounds = config['y_bounds']
    # Extrai o tamanho da figura (largura, altura)
    fig_size = config['fig_size']
    # Extrai a lista de paredes do mapa
    walls = config['walls']
    
    return x_bounds, y_bounds, fig_size, walls

# Função principal que cria o mapa visual
def make_map(close=True):
    # Carrega as configurações do mapa
    x_bounds, y_bounds, fig_size, walls = load_map()
    
    # Cria uma nova figura (tela para desenhar) com o tamanho especificado
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Define os limites do eixo X (horizontal)
    ax.set_xlim(x_bounds)
    # Define os limites do eixo Y (vertical)
    ax.set_ylim(y_bounds)
    # Mantém a proporção igual para que o mapa não fique distorcido
    ax.set_aspect('equal')
    
    # Para cada parede na lista de paredes
    for wall in walls:
        # Extrai as coordenadas dos dois pontos que definem a parede
        (x1, y1), (x2, y2) = wall
        # Desenha uma linha entre os dois pontos:
        # 'k-' = linha preta contínua, lw=2 = largura da linha 2
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2)
    
    # Remove os eixos (números e bordas) da visualização
    plt.axis('off')
    # Fecha a figura para que não seja mostrada imediatamente
    if close:
        plt.close()
    
    # Retorna a figura criada para que possa ser usada depois
    return fig, ax