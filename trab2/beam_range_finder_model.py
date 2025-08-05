import numpy as np        
import math               
import matplotlib.pyplot as plt  

from lidar_ray_casting import simulate_lidar,get_beam_angles,get_pose
from world import get_world,load_map,make_map
# -----------------------------------------------------------------------------
# FUNÇÕES DE PROBABILIDADE QUE DEFINEM OS QUATRO MODELOS DE RUÍDO DO SENSOR
# -----------------------------------------------------------------------------

def p_hit(z, z_star, sigma_hit, max_range):
    """
    Modelo gaussiano (p_hit):
    Representa a probabilidade de a medição ser próxima do valor esperado (ruído normal).
    z       : valor medido pelo sensor (distância)
    z_star  : valor esperado (distância "ideal" calculada via ray casting)
    sigma_hit : desvio padrão do ruído gaussiano (quanto maior, mais "espalhada" a curva)
    max_range : alcance máximo do sensor (trunca valores fora desse intervalo)
    """
    if 0 <= z <= max_range:
        # Fórmula da distribuição normal (Gaussiana) centrada em z_star
        return (1 / (math.sqrt(2 * math.pi) * sigma_hit)) * math.exp(-0.5 * ((z - z_star) / sigma_hit)**2)
    return 0.0  # Se a medição for negativa ou maior que o alcance máximo, probabilidade é zero


def p_short(z, z_star, lambda_short):
    """
    Modelo exponencial (p_short):
    Representa a probabilidade de leituras menores do que o esperado (obstáculos inesperados).
    z       : valor medido
    z_star  : valor esperado (distância sem obstáculos extras)
    lambda_short : taxa da exponencial (quanto maior, mais rápido decai a probabilidade)
    """
    if 0 <= z <= z_star:
        # Fator de normalização para garantir que a integral de 0 a z_star seja 1
        eta = 1 - math.exp(-lambda_short * z_star)
        return (lambda_short * math.exp(-lambda_short * z)) / eta
    return 0.0  # Leituras maiores que o esperado não entram nesse modelo


def p_max(z, max_range):
    """
    Modelo p_max:
    Probabilidade associada a medições iguais ao valor máximo do sensor (sem detecção de obstáculo).
    z        : valor medido
    max_range: alcance máximo do sensor
    """
    return 1.0 if z == max_range else 0.0


def p_rand(z, max_range):
    """
    Modelo p_rand:
    Modelo aleatório uniforme para ruído completamente aleatório.
    z        : valor medido
    max_range: alcance máximo do sensor
    """
    if 0 <= z <= max_range:
        # Probabilidade uniforme é 1 dividido pelo alcance total
        return 1.0 / max_range
    return 0.0

# -----------------------------------------------------------------------------
# MODELO PRINCIPAL: BEAM RANGE FINDER MODEL
# -----------------------------------------------------------------------------

def beam_range_finder_model(zt, pose, walls_world, params,beam_angles):
    """
    Calcula a probabilidade P(zt | pose, mapa) usando o modelo beam_range_finder.
    - zt: lista de leituras do sensor (uma distância para cada feixe)
    - pose: tupla (x, y, theta) representando a posição e orientação do robô
    - walls_world: lista de paredes no mapa, cada parede é ((x1,y1), (x2,y2))
    - params: dicionário com todos os parâmetros do modelo probabilístico
    """
    # 1. Faz ray casting para calcular as distâncias esperadas (sem ruído)
    # simulate_lidar retorna duas listas: distâncias e pontos de interseção
    z_expected, _ = simulate_lidar(pose, beam_angles, walls_world, max_range=params["max_range"])
    
    # 2. Inicializa a probabilidade total como 1 (vamos multiplicar cada feixe depois)
    q = 1.0
    
    # 3. Para cada feixe (medição real z e valor esperado z_star)
    for z, z_star in zip(zt, z_expected):
        # Calcula a probabilidade combinada do feixe usando mistura dos quatro modelos
        p = (params["z_hit"]   * p_hit(z, z_star, params["sigma_hit"], params["max_range"]) +
             params["z_short"] * p_short(z, z_star, params["lambda_short"]) +
             params["z_max"]   * p_max(z, params["max_range"]) +
             params["z_rand"]  * p_rand(z, params["max_range"]))
        # Multiplica na probabilidade total
        q *= p

    # 4. Retorna a probabilidade total (likelihood) para essa pose
    return q

# -----------------------------------------------------------------------------
# FUNÇÃO PARA SIMULAR EM TODA A GRADE DO MAPA
# -----------------------------------------------------------------------------

def simulate_beam_model_map(walls_world, measurements, params,beam_angles, grid_res=0.2):
    """
    Simula o beam_range_finder_model sobre uma grade de posições (x,y) no mapa.
    - walls_world: paredes do mapa
    - measurements: leituras simuladas do sensor (com ruído)
    - params: parâmetros do modelo
    - grid_res: resolução da grade (quanto menor, mais pontos avaliados)
    Retorna: 
    xs, ys -> vetores com coordenadas x e y avaliadas
    prob_grid -> matriz 2D com probabilidades normalizadas
    """
    # 1. Define os limites do mapa (nesse caso fixamos de 0 a 10)
    xs = np.arange(0, 10+grid_res, grid_res)  # coordenadas no eixo x
    ys = np.arange(0, 10+grid_res, grid_res)  # coordenadas no eixo y

    # 2. Cria uma matriz 2D para armazenar a probabilidade de cada ponto
    prob_grid = np.zeros((len(ys), len(xs)))
    
    # 3. Mantemos o ângulo fixo (poderíamos testar vários ângulos depois)
    theta = np.pi/4

    # 4. Percorre cada ponto da grade (duplo for: primeiro Y, depois X)
    for iy, y in enumerate(ys):       # iy = índice da linha, y = valor real de y
        for ix, x in enumerate(xs):   # ix = índice da coluna, x = valor real de x
            # Cria a pose (x, y, theta) para avaliar
            pose = (x, y, theta)
            # Calcula a probabilidade para essa pose e armazena na matriz
            prob_grid[iy, ix] = beam_range_finder_model(measurements, pose, walls_world, params,beam_angles)

    # 5. Normaliza os valores da matriz, transformando em probabilidade
    prob_grid /= np.sum(prob_grid)
    return xs, ys, prob_grid

# -----------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL PARA TESTAR E PLOTAR
# -----------------------------------------------------------------------------

def main():

    fig,ax = make_map(False)
    (x,y,theta) = get_pose()
    plt.plot(x,y, 'bo', label='Pose real')

    # Círculo representando o robô
    robot = plt.Circle((x, y), 0.3, color='blue', alpha=0.5)
    ax.add_patch(robot)

    # Seta indicando orientação
    ax.arrow(
        x, y,
        0.5 * np.cos(theta),  # deslocamento x da seta
        0.5 * np.sin(theta),  # deslocamento y da seta
        head_width=0.1,
        color='blue'
    )
    
    _,_,fig_size,_ = load_map()
    
    walls_world = get_world()

    # 1. Simula leituras reais do sensor (sem ruído) e adiciona ruído gaussiano
    true_dists, _ = simulate_lidar(get_pose(), get_beam_angles(), walls_world)
    noisy_measurements = [d + np.random.normal(0, 0.1) for d in true_dists]  # ruído com desvio 0.1m

    # 4. Define os parâmetros do modelo probabilístico
    # -----------------------------------------------------------------------------
    # Esses parâmetros controlam como o modelo beam_range_finder interpreta as leituras do sensor.
    # Eles representam pesos e características estatísticas que misturam quatro modelos diferentes
    # de ruído: p_hit, p_short, p_max e p_rand. A soma dos pesos z_hit, z_short, z_max e z_rand
    # deve ser 1.0, pois eles formam uma combinação convexa (mistura de distribuições).
    #
    # A ideia: cada feixe do LIDAR pode vir de quatro "causas" possíveis:
    # - p_hit   : o feixe detectou corretamente a parede esperada, com ruído gaussiano.
    # - p_short : o feixe foi bloqueado por um obstáculo inesperado (ex.: gato), leitura mais curta.
    # - p_max   : o feixe não encontrou obstáculo e retornou o valor máximo do sensor.
    # - p_rand  : o feixe deu uma leitura completamente aleatória (ruído puro).
    #
    # Alterar esses valores muda como o modelo "confia" nas leituras. Abaixo, explicamos um a um:
    #
    # z_hit (0.6):
    #   - Peso do modelo gaussiano (p_hit).
    #   - Representa a fração de leituras que seguem o comportamento "ideal" (detecção correta com ruído pequeno).
    #   - Maior valor → mais confiança no mapa, resultados mais focados.
    #   - Menor valor → modelo assume mais ruído, probabilidade se espalha.
    #
    # z_short (0.1):
    #   - Peso do modelo de leituras curtas (p_short).
    #   - Captura situações em que o feixe é interrompido antes do esperado por objetos não mapeados.
    #   - Maior valor → modelo lida melhor com obstáculos inesperados.
    #   - Menor valor → modelo ignora quase todas as leituras curtas.
    #
    # z_max (0.1):
    #   - Peso do modelo de leituras máximas (p_max).
    #   - Corresponde a casos em que o feixe alcança o limite máximo do sensor (sem encontrar nada).
    #   - Maior valor → modelo aceita mais leituras "infinito".
    #   - Menor valor → modelo espera sempre encontrar paredes (descarta leituras máximas).
    #
    # z_rand (0.1):
    #   - Peso do modelo de leituras aleatórias (p_rand).
    #   - Trata leituras totalmente erráticas como possíveis.
    #   - Maior valor → modelo tolera medições absurdas ou reflexos.
    #   - Menor valor → medições aleatórias derrubam a probabilidade da pose.
    #
    # sigma_hit (0.5):
    #   - Desvio padrão do ruído gaussiano usado em p_hit.
    #   - Controla o "quanto" de variação aceitaremos em torno da distância esperada.
    #   - Menor valor → curva gaussiana mais "pontuda", modelo mais exigente.
    #   - Maior valor → curva mais larga, modelo mais tolerante.
    #
    # lambda_short (1.0):
    #   - Taxa do decaimento exponencial em p_short.
    #   - Define quão rápido a probabilidade cai para leituras curtas.
    #   - Maior valor → só aceita leituras curtas muito próximas de zero.
    #   - Menor valor → aceita leituras curtas mais distantes do esperado.
    #
    # max_range (20.0):
    #   - Alcance máximo do sensor LIDAR.
    #   - Usado para p_max e para limitar os outros cálculos.
    #   - Se muito baixo → truncará leituras válidas.
    #   - Se muito alto → aumenta influência do modelo aleatório p_rand.
    # -----------------------------------------------------------------------------
    params = {
        "z_hit": 0.6,
        "z_short": 0.1,
        "z_max": 0.1,
        "z_rand": 0.1,
        "sigma_hit": 0.5,
        "lambda_short": 1.0,
        "max_range": 20.0
    }

    # 5. Executa simulação: calcula probabilidade para cada ponto do mapa
    xs, ys, prob_grid = simulate_beam_model_map(walls_world, noisy_measurements, params,get_beam_angles())

    # 7. Plota o resultado como um "heatmap" (mapa de calor)
    
    plt.figure(figsize=fig_size)
    # Mostra probabilidade (0-1) como cores; extent define limites reais em metros
    plt.imshow(prob_grid, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower', cmap='hot')
    plt.colorbar(label="Probabilidade normalizada")
    # Marca a posição real do robô em azul
    plt.plot(get_pose()[0], get_pose()[1], 'bo', label='Pose real')
    plt.legend()
    plt.title("Mapa de probabilidade - Beam Range Finder Model")
    plt.show()
if __name__ == "__main__":
    main()
