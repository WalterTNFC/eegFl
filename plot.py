import matplotlib.pyplot as plt
import pandas as pd

# Dados de entrada (representando a tabela fornecida)
data = [
    # Round 1
    [1, 0.25, 'client_1'], [2, 0.25, 'client_1'], [3, 0.25, 'client_1'], [4, 0.25, 'client_1'],
    [5, 0.25, 'client_1'], [6, 0.25, 'client_1'], [7, 0.25, 'client_1'], [8, 0.25, 'client_1'],
    [9, 0.25, 'client_1'], [10, 0.25, 'client_1'], [11, 0.25, 'client_1'], [12, 0.25, 'client_1'],
    [13, 0.25, 'client_1'], [14, 0.25, 'client_1'], [15, 0.2604, 'client_1'], [16, 0.2604, 'client_1'],
    [17, 0.2604, 'client_1'], [18, 0.2708, 'client_1'], [19, 0.2708, 'client_1'], [20, 0.2812, 'client_1'],
    [21, 0.3125, 'client_1'], [22, 0.3229, 'client_1'], [23, 0.3333, 'client_1'], [24, 0.3542, 'client_1'],
    [25, 0.3958, 'client_1'],
    # Round 2
    [1, 2.500, 'client_2'], [2, 2.500, 'client_2'], [3, 2.500, 'client_2'], [4, 2.500, 'client_2'],
    [5, 2.708, 'client_2'], [6, 3.333, 'client_2'], [7, 3.958, 'client_2'], [8, 4.271, 'client_2'],
    [9, 4.479, 'client_2'], [10, 4.583, 'client_2'], [11, 4.583, 'client_2'], [12, 4.688, 'client_2'],
    [13, 4.688, 'client_2'], [14, 4.688, 'client_2'], [15, 4.688, 'client_2'], [16, 4.688, 'client_2'],
    [17, 4.688, 'client_2'], [18, 4.688, 'client_2'], [19, 4.688, 'client_2'], [20, 4.688, 'client_2'],
    [21, 4.688, 'client_2'], [22, 4.688, 'client_2'], [23, 4.688, 'client_2'], [24, 4.688, 'client_2'],
    [25, 4.688, 'client_2'],
    # Round 3
    [1, 2.917, 'client_3'], [2, 2.500, 'client_3'], [3, 2.500, 'client_3'], [4, 2.500, 'client_3'],
    [5, 2.500, 'client_3'], [6, 2.708, 'client_3'], [7, 3.750, 'client_3'], [8, 4.583, 'client_3'],
    [9, 4.583, 'client_3'], [10, 3.958, 'client_3'], [11, 3.854, 'client_3'], [12, 3.438, 'client_3'],
    [13, 3.438, 'client_3'], [14, 3.333, 'client_3'], [15, 3.333, 'client_3'], [16, 3.333, 'client_3'],
    [17, 3.438, 'client_3'], [18, 3.542, 'client_3'], [19, 3.646, 'client_3'], [20, 3.646, 'client_3'],
    [21, 3.750, 'client_3'], [22, 3.854, 'client_3'], [23, 4.062, 'client_3'], [24, 4.271, 'client_3'],
    [25, 4.375, 'client_3'],
    # Round 4
    [1, 2.500, 'client_4'], [2, 2.500, 'client_4'], [3, 2.500, 'client_4'], [4, 2.708, 'client_4'],
    [5, 3.646, 'client_4'], [6, 4.583, 'client_4'], [7, 4.688, 'client_4'], [8, 4.688, 'client_4'],
    [9, 4.583, 'client_4'], [10, 4.375, 'client_4'], [11, 4.375, 'client_4'], [12, 4.479, 'client_4'],
    [13, 4.583, 'client_4'], [14, 4.792, 'client_4'], [15, 4.896, 'client_4'], [16, 4.896, 'client_4'],
    [17, 5.000, 'client_4'], [18, 5.000, 'client_4'], [19, 5.000, 'client_4'], [20, 5.000, 'client_4'],
    [21, 5.104, 'client_4'], [22, 5.208, 'client_4'], [23, 5.208, 'client_4'], [24, 5.208, 'client_4'],
    [25, 5.312, 'client_4'],
    # Round 5
    [1, 2.500, 'client_5'], [2, 2.500, 'client_5'], [3, 2.500, 'client_5'], [4, 2.500, 'client_5'],
    [5, 2.500, 'client_5'], [6, 2.500, 'client_5'], [7, 2.500, 'client_5'], [8, 2.812, 'client_5'],
    [9, 3.542, 'client_5'], [10, 4.219, 'client_5'], [11, 4.844, 'client_5'], [12, 6.042, 'client_5'],
    [13, 7.500, 'client_5'], [14, 8.333, 'client_5'], [15, 8.854, 'client_5'], [16, 9.115, 'client_5'],
    [17, 9.323, 'client_5'], [18, 9.531, 'client_5'], [19, 9.688, 'client_5'], [20, 9.688, 'client_5'],
    [21, 9.740, 'client_5'], [22, 9.740, 'client_5'], [23, 9.792, 'client_5'], [24, 9.792, 'client_5'],
    [25, 9.792, 'client_5']
]

# Converte os dados em um DataFrame
df = pd.DataFrame(data, columns=['epoch', 'train_accuracy', 'round'])

# Criação do gráfico para as cinco curvas
plt.figure(figsize=(8, 6))  # Diminui o tamanho da figura

# Plotando as curvas para cada round
for client_name in df['round'].unique():
    client_data = df[df['round'] == client_name]
    plt.plot(client_data['epoch'], client_data['train_accuracy'], label=client_name, marker='o')

# Adicionando título, legendas e rótulos
plt.title('Acurácia por Época para Cada Cliente', fontsize=14)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Acurácia', fontsize=12)
plt.legend(title='Clientes', fontsize=10, title_fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid mais detalhado
plt.minorticks_on()  # Ativa as marcas menores do grid
plt.grid(True, which='minor', linestyle=':', linewidth=0.5)  # Grid mais detalhado

# Exibindo o gráfico
plt.show()
