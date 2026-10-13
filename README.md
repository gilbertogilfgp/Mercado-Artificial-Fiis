# **FiiLMA**
### *Laboratório de Mercado Artificial para Fundos Imobiliários*

Este repositório contém a implementação de um mercado artificial de Fundos de Investimento Imobiliário (FIIs) baseado em agentes heterogêneos, com microestrutura de mercado via order book com leilão duplo contínuo.

## Objetivo
Desenvolver um modelo computacional capaz de simular a dinâmica de preços, dividendos, sentimento, negociação e efeitos de choques em um mercado artificial de FIIs.

## Estrutura inicial
- `agentes.py`: definição dos agentes investidores
- `ativos.py`: definição dos imóveis e do FII
- `microestrutura.py`: ordens, transações e order book
- `mercado.py`: ambiente de mercado e processamento diário
- `simulacao.py`: montagem e execução da simulação
- `notebooks/`: testes e análises exploratórias
- `results/`: saídas da simulação

## Status
Projeto em desenvolvimento.


## Como Executar no Google Colab

Este projeto pode ser executado diretamente no Google Colab a partir do repositório no GitHub.

### 1. Clonar o repositório

```python
!git clone https://github.com/gilbertogilfgp/Mercado-Artificial-Fiis.git
```

### 2. Entrar na pasta do projeto

```python
%cd /content/Mercado-Artificial-Fiis
```

### 3. Adicionar a raiz do projeto ao caminho do Python

```python
import sys
sys.path.append('/content/Mercado-Artificial-Fiis')
```

### 4. Instalar as dependências

```python
!pip install -r requirements.txt
```

Caso o arquivo `requirements.txt` ainda não esteja completo, pode-se instalar manualmente:


### 5. Importar a função principal da simulação

```python
from src.simulacao import simular_mercado_e_plotar
```

### 6. Definir os parâmetros do sistema

```python
parametros_sistema = [0.848228, 0.695407, 0.878559, 0.155958, 0.075485]
params = copy.deepcopy(SIM_PARAMS)
NUM_DIAS    = 60
```
### 7. Configurar choques se desejar
```params = copy.deepcopy(SIM_PARAMS)

# Opcional: isola os choques programados (desliga o ruído aleatório diário).
# Deixe comentado para manter o mercado com ruído de fundo.
params["prob_choque_diario"] = 0.0025

# Um choque de cada categoria. Ajuste dias, sinais e magnitudes à vontade.
params["choques"] = [
    # Notícia (dissipa ao longo da duração)
    {"categoria": "noticia", "dia": 25, "tipo": "positivo",
     "intensidade": 1.0, "duracao": 5, "delta": 1},
    
    {"categoria": "noticia", "dia": 30, "tipo": "positivo",
     "intensidade": 1.0, "duracao": 5, "delta": 1},
    
    {"categoria": "noticia", "dia": 45, "tipo": "negativo",
     "intensidade": 1, "duracao": 5, "delta": 1},

    # Macro (muda o Banco Central a partir do dia; base: inflação 0.07, prêmio 0.08)
    #{"categoria": "macro", "dia": 30, "inflacao": 0.10, "premio": 0.11},

    # Micro (muda vacância e custo dos imóveis, em %, a partir do dia)
    #{"categoria": "micro", "dia": 45, "vac": 30, "custo": 30},
]
```

### 8. Executar uma simulação de teste

```python
resultado = simular_mercado_e_plotar(
    parametros_sistema=parametros_sistema,
    num_dias=60,
    imprimir=False
)
```
### 9. Plotar
```python

plotar_resultado(resultado, params, num_dias=60)
```

## Bloco único para execução no Colab

```python
import numpy as np
import matplotlib.pyplot as plt
import copy

!git clone https://github.com/gilbertogilfgp/Mercado-Artificial-Fiis.git
%cd /content/Mercado-Artificial-Fiis

import sys, types

src_package = types.ModuleType('src')
src_package.__path__ = ['/content/Mercado-Artificial-Fiis/src']
src_package.__package__ = 'src'
sys.modules['src'] = src_package

for mod in list(sys.modules.keys()):
    if mod.startswith('src.'):
        del sys.modules[mod]

if '/content/Mercado-Artificial-Fiis' not in sys.path:
    sys.path.insert(0, '/content/Mercado-Artificial-Fiis')

!pip install -r requirements.txt --quiet


from src.simulacao import simular_mercado_e_plotar, plotar_resultado, SIM_PARAMS

parametros_sistema = [0.848228, 0.695407, 0.878559, 0.155958, 0.075485]
params = copy.deepcopy(SIM_PARAMS)
NUM_DIAS    = 60

# Simular Choques
params = copy.deepcopy(SIM_PARAMS)

# Opcional: isola os choques programados (desliga o ruído aleatório diário).
# Deixe comentado para manter o mercado com ruído de fundo.
# params["prob_choque_diario"] = 0.00

# Um choque de cada categoria. Ajuste dias, sinais e magnitudes à vontade.
params["choques"] = [
    # Notícia (dissipa ao longo da duração)
    {"categoria": "noticia", "dia": 25, "tipo": "positivo",
     "intensidade": 1.0, "duracao": 5, "delta": 1},
    
    {"categoria": "noticia", "dia": 30, "tipo": "positivo",
     "intensidade": 1.0, "duracao": 5, "delta": 1},
    
    {"categoria": "noticia", "dia": 45, "tipo": "negativo",
     "intensidade": 1, "duracao": 5, "delta": 1},

    # Macro (muda o Banco Central a partir do dia; base: inflação 0.07, prêmio 0.08)
    #{"categoria": "macro", "dia": 30, "inflacao": 0.10, "premio": 0.11},

    # Micro (muda vacância e custo dos imóveis, em %, a partir do dia)
    #{"categoria": "micro", "dia": 45, "vac": 30, "custo": 30},
]

# Rodar sem plotar
resultado = simular_mercado_e_plotar(
    parametros_sistema=parametros_sistema,
    num_dias=NUM_DIAS,
    sim_params=params,
    imprimir=False,        
)


# Plotar quando quiser, com choques destacados
plotar_resultado(resultado, params, num_dias=60)
```

## Observações

* Como o projeto utiliza uma estrutura com pasta `src`, é necessário adicionar a raiz do repositório ao `sys.path`.
* Recomenda-se começar com um número pequeno de dias, como `10` ou `30`, para validar a execução inicial.
* Após confirmar o funcionamento, o número de dias pode ser ampliado para experimentos mais longos.
