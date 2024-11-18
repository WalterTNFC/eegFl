## Inspiração

Artigo: "Aggregating intrinsic information to enhance BCI performance through
federated learning"

- **Desafio de Dados Insuficientes em BCI**: 
  - A falta de dados é um desafio antigo para construir modelos de aprendizado profundo de alto desempenho em Interfaces Cérebro-Computador (BCI).
  - Mesmo com diversos grupos coletando conjuntos de dados de EEG para a mesma tarefa, o compartilhamento de dados entre dispositivos é difícil devido à heterogeneidade dos dispositivos.

- **Importância da Diversidade de Dados**:
  - A diversidade de dados é fundamental para a robustez dos modelos, mas poucos trabalhos abordam essa questão, concentrando-se principalmente no treinamento de modelos dentro de um único conjunto de dados.

- **Proposta de Solução**:
  - **FLEEG (Federated Learning EEG Decoding)**: Estrutura hierárquica personalizada de aprendizado federado para superar o desafio da diversidade de dados.
  - Permite que conjuntos de dados com formatos diferentes colaborem no processo de treinamento do modelo.

- **Funcionamento do Framework**:
  - Cada cliente recebe um conjunto de dados específico e treina um modelo hierárquico personalizado.
  - O servidor coordena o treinamento para compartilhar conhecimento entre todos os conjuntos de dados, melhorando o desempenho geral.

- **Contribuição Inovadora**:
  - Primeira solução end-to-end para enfrentar o desafio de heterogeneidade nos dados de EEG em BCI.

![FLEEG](./figures/Captura%20de%20tela%20de%202024-11-17%2016-49-01.png)

## Proposta

Federar modelo proposto em: [**How to train, test and tune your model?**](https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html#sphx-glr-auto-examples-model-building-plot-how-train-test-and-tune-py). Este tutorial mostra como treinar, ajustar e testar adequadamente seus modelos de aprendizado profundo com o [**Braindecode**](https://braindecode.org/stable/index.html). Usaremos o conjunto de dados [**BCIC IV 2a**](https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html#id5) como exemplo de demonstração. **Os eventos considerados nos dados são apenas as 4 classes alvo (esquerda, direita, pé, língua).**

### Braindecode
![BrainDeCode](./figures/brainDeCode.png)


O Braindecode é uma ferramenta open-source em Python para decodificação de dados cerebrais EEG com modelos de aprendizado profundo. Aplicações:
-   Pré-processamento
-   Visualização de dados
-   Implementações de várias arquiteturas de aprendizado profundo

Assim, o dataset será dividido em cinco partes, e cada uma delas passará por um pré-processamento específico. Em seguida, cada parte será alocada para um cliente diferente, com o objetivo de verificar como o aprendizado federado pode melhorar a performance e a capacidade de generalização do modelo aplicado.

## Prática
### Instalação de bibliotecas
```python
!pip install braindecode moabb ray
!pip install -U "flwr[simulation]"
```
### Carregar o dataset
```python
from braindecode.datasets import MOABBDataset
from sklearn.model_selection import train_test_split
```

```python
import numpy as np
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)

# Carregar o dataset
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])

# Dividir o dataset em 5 partes manualmente
num_parts = 5
part_size = len(dataset.datasets) // num_parts
dataset_parts = []

# Separar manualmente os dados e recriar subconjuntos como novos datasets
for i in range(num_parts):
    start = i * part_size
    end = (i + 1) * part_size if i < num_parts - 1 else len(dataset.datasets)
    subset = dataset.datasets[start:end]
    dataset_parts.append(subset)

# Definir diferentes conjuntos de parâmetros para cada parte
param_sets = [
    {"low_cut_hz": 4.0, "high_cut_hz": 30.0, "factor_new": 1e-3, "init_block_size": 1000},
    {"low_cut_hz": 5.0, "high_cut_hz": 35.0, "factor_new": 5e-4, "init_block_size": 1200},
    {"low_cut_hz": 3.5, "high_cut_hz": 28.0, "factor_new": 2e-3, "init_block_size": 900},
    {"low_cut_hz": 6.0, "high_cut_hz": 40.0, "factor_new": 1e-4, "init_block_size": 1500},
    {"low_cut_hz": 4.5, "high_cut_hz": 32.0, "factor_new": 1e-3, "init_block_size": 1100},
]

# Lista para armazenar os datasets processados
processed_datasets = []

# Aplicar os preprocessadores a cada subconjunto com parâmetros específicos
for i, (subset, params) in enumerate(zip(dataset_parts, param_sets)):
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data, factor: np.multiply(data, factor), factor=1e6),
        Preprocessor("filter", l_freq=params["low_cut_hz"], h_freq=params["high_cut_hz"]),
        Preprocessor(exponential_moving_standardize, factor_new=params["factor_new"], init_block_size=params["init_block_size"]),
    ]

    # Criar um novo dataset a partir do subconjunto
    new_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
    new_dataset.datasets = subset  # Atribuir subconjunto ao novo dataset

    # Preprocessar o novo dataset de forma independente
    preprocess(new_dataset, preprocessors, n_jobs=-1)
    print(f"Preprocessamento concluído para o subconjunto {i+1} com parâmetros: {params}")

    # Armazenar o dataset processado na lista
    processed_datasets.append(new_dataset)
```

### Criação de janelas de tempo

A função ***extractionWindow*** tem o objetivo criar "janelas" que correspondem aos eventos de interesse no sinal de EEG.

```python
def extractionWindow(dataset):
  # A janela começa a ser registrada meio segundo antes do evento de interesse 
  trial_start_offset_seconds = -0.5
  # Extração de frequência do conjunto de dados de entrada
  sfreq = dataset.datasets[0].raw.info["sfreq"]
  # Verificação da frequencia
  assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
  # Deslocamento em amostras para o início da janela de dados
  trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

  # Divide o conjunto de dados em janelas de tempo, com base em eventos específicos
  windows_dataset = create_windows_from_events(
      dataset,
      trial_start_offset_samples=trial_start_offset_samples,
      trial_stop_offset_samples=0,
      preload=True,
  )


  return windows_dataset
```

### Divisão dos dados:
```python
def split_windows_dataset(windows_dataset):
  splitted = windows_dataset.split("session")
  if('0train' in splitted):
    return splitted['0train']
  else:
    return splitted['1test']
```

### Treinamento do modelo:
[ShallowNet](https://www.researchgate.net/figure/The-Shallow-Convolutional-Network-architecture-proposed-for-the-BCI-Competition-IV_fig2_355779357)
```python
    from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet

def create_model(shape):
  seed = 20200220

  n_classes = 4
  classes = list(range(n_classes))
  # Extract number of chans and time steps from dataset
  n_channels = shape[0]
  input_window_samples = shape[1]

  model = ShallowFBCSPNet(
      n_channels,
      n_classes,
      input_window_samples=input_window_samples,
      final_conv_length="auto",
  )

  return model
```

### Instanciando os clientes:

Mapear cada Id de cliente para um dataset específico
```python 
    def get_client_dataset(id):
    client_datasets = {
        0: processed_datasets[0],
        1: processed_datasets[1],
        2: processed_datasets[2],
        3: processed_datasets[3],
        4: processed_datasets[4]
    }
    return client_datasets.get(id)
```

Cria uma instância do cliente federado com o modelo e o dataset atribuídos ao cliente.

```python
def numpyclient_fn(context):
    # Usando a função de mapeamento para pegar os dados baseados no client_id do context
    client_id = context.node_config["partition-id"]
    dataset_to_use = get_client_dataset(client_id)

    windows_dataset = extractionWindow(dataset_to_use)
    model = create_model(windows_dataset[0][0].shape)
    # Continua a configuração do cliente
    client = FlowerNumPyClient(model, windows_dataset)
    return client.to_client()
```

-   get_parameters: Extrai os parâmetros do modelo como uma lista de arrays NumPy.
-   set_parameters: Restaura os parâmetros do modelo a partir de uma lista de arrays, convertendo-os de volta para tensores PyTorch e mantendo a associação com os nomes dos parâmetros.
```python
from collections import OrderedDict
import torch

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
```

```python
from skorch.callbacks import LRScheduler
from braindecode import EEGClassifier
import torch
from collections import OrderedDict

class FlowerNumPyClient(fl.client.NumPyClient):
    # Constantes de hiperparâmetros
    LR = 0.0625 * 0.01
    WEIGHT_DECAY = 0
    BATCH_SIZE = 64
    N_EPOCHS = 25

    def __init__(self, model, windows_dataset):
        self.model = model
        self.window = windows_dataset
        self.train_set = split_windows_dataset(self.window)

        print('Inicializando o classificador')
        self.clf = self._initialize_classifier()

    def _initialize_classifier(self):
        return EEGClassifier(
            self.model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=None,
            optimizer__lr=self.LR,
            optimizer__weight_decay=self.WEIGHT_DECAY,
            batch_size=self.BATCH_SIZE,
            callbacks=[
                "accuracy",
                ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=self.N_EPOCHS - 1)),
            ],
            device='cpu',
            classes=list(range(4)),  # 4 classes no dataset
            max_epochs=self.N_EPOCHS,
        )

    def get_parameters(self, config):
        print('Obtendo parâmetros do modelo')
        return get_parameters(self.model)

    def set_weights(self, net, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    # Treinamento
    def fit(self, parameters, config):
        print('Iniciando o treinamento (fit)')
        self.set_weights(self.model, parameters)
        
        # Treinamento do classificador
        self.clf.fit(self.train_set, y=None)

        return self.get_parameters(self.model), len(self.window), {}

    # Avaliação
    def evaluate(self, parameters, config):
        print('Iniciando a avaliação')
        self.set_weights(self.model, parameters)
        
        # Avaliação do modelo após o treinamento
        y_test = self.train_set.get_metadata().target
        test_acc = self.clf.score(self.train_set, y=y_test)
        print(f"Test acc: {(test_acc * 100):.2f}%")
        
        return float(test_acc), len(self.train_set), {"accuracy": float(test_acc)}
```

### Definição do meu servidor
```python
def server_fn(context):
    config = fl.server.ServerConfig(num_rounds=1)
    return fl.server.ServerAppComponents(config=config)


# Create ServerApp
server = fl.server.ServerApp(server_fn=server_fn)
```

### Iniciar a simulação
```python
NUM_PARTITIONS = 5

# Run simulation
# Flower ClientApp
app = fl.client.ClientApp(numpyclient_fn)
fl.simulation.run_simulation(
    server_app=server,
    client_app= app,
    num_supernodes=NUM_PARTITIONS,
)

``````
### Resultado
![Resultado](./figures/Figure_1.png)