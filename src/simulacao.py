import numpy as np
import matplotlib.pyplot as plt

from src.ativos import Imovel, FII
from src.agentes import Agente, gerar_literacia_financeira
from src.mercado import BancoCentral, Midia, Mercado

def calcular_sentimento_medio(agentes):
    return np.mean([agente.sentimento for agente in agentes])



def simular_mercado_e_plotar(parametros_sistema, num_dias, imprimir = False):
    random_seed = 42
    # random.seed(random_seed)
    # np.random.seed(random_seed)

    sim_params = {
        # ... (seus parâmetros)
        "num_dias": num_dias, # Ajustei o número de dias para ser testável. Você pode aumentar.
        "total_cota": 100_000,

        # Parâmetros do FII
        "fii": {
            "num_cotas": 100_000,
            "caixa_inicial": 50_000,
            "dividendos_taxa": 0.95,          # Fração do fluxo destinada à distribuição de dividendos
            "dividendos_caixa_taxa": 0.05,      # Fração que permanece no caixa
            "investimento_fracao": 0.50,        # Fração do caixa a ser investida na atualização dos imóveis
            "memoria": True,

        },

        # Parâmetros dos Imóveis (cada dicionário pode ter parâmetros específicos para o imóvel)
        "imoveis": [
            {"valor": 1_000_000, "vacancia": 0.1, "custo_manutencao": 200,
             "params": {"aluguel_factor": 0.005, "desvio_normal": 0.01}},
            {"valor": 2_000_000, "vacancia": 0.2, "custo_manutencao": 500,
             "params": {"aluguel_factor": 0.005, "desvio_normal": 0.01}},
        ],

        # Parâmetros dos Agentes
        "num_agentes_pf": 600,
        "prop_cota_agente": 0.6,
        "agente_pf": {
            "caixa_inicial": 10_000,
            "cotas_iniciais_primeiro": 100,
            "cotas_iniciais_outros": 100,
            "num_vizinhos": 30,
            "literacia_media": 0.3,
            "literacia_std": 0.4,
            "expectativa_inflacao": 0.05,
            "expectativa_premio": 0.08,
            # Parâmetros específicos do Agente (para cálculos internos)
            "params": {
                "window_chart": 21,
                "alpha_chart_short": 0.3,
                "alpha_chart_long": 0.1,
                "ruido_std": 0.1,
                "peso_retorno": 0.6,
                "peso_riqueza": 0.4,
            }
        },

        # Parâmetros dos Agentes PJ
        "num_agentes_pj": 200,
        "prop_cota_agente_pj": 0.2,
        "agente_pj": {
            "caixa_inicial": 10_000,
            "cotas_iniciais_primeiro": 100,
            "cotas_iniciais_outros": 100,
            "num_vizinhos": 30,
            "literacia_media": 0.3,
            "literacia_std": 0.4,
            "expectativa_inflacao": 0.05,
            "expectativa_premio": 0.08,
            # Parâmetros específicos do Agente PJ
            "params": {
                "window_chart": 21,
                "alpha_chart_short": 0.3,
                "alpha_chart_long": 0.1,
                "ruido_std": 0.05,
                "peso_retorno": 0.3,
                "peso_riqueza": 0.7,
            }
        },

        # Parâmetros do Banco Central
        "banco_central": {
            "taxa_selic": 0.15,
            "expectativa_inflacao": 0.07,
            "premio_risco": 0.08,
        },

        # Parâmetros da Mídia
        "midia": {
            "valor_inicial": 0,
            "sigma": 0.1,
            "valores_fixos": {#10: 1, 20: 0, 30: -3, 40: 0,
                #50: 2, 100: -3, 101: -2, 150: 0, 200: 1, 250: 1,
                #300: -3, 350: 0, 400: 0, 450: 0, 500: 0,
                #550: 1, 600: -3, 610: -3, 620: -2,
                #700: 0, 800: 0, 850: -1, 900: 0, 950: 1,
                #1000: -1, 1050: 0, 1100: 1, 1150: 1, 1200: 0, 1250: 1
            }
        },

        # Parâmetros para o sentimento (utilizados na criação de ordens e nos cálculos dos agentes)
        "parametros_sentimento": {
            "a0": parametros_sistema[0],
            "b0": parametros_sistema[1],
            "c0": parametros_sistema[2],
            "beta": parametros_sistema[3],
            "peso_preco_esperado": parametros_sistema[4],
            "ruido_std":0.1 ,
            "sigma_midia": 0.8,
            "piso_prob_negociar": 0.1,
            "peso_sentimento_inflacao": 0.4,
            "peso_sentimento_expectativa": 0.4,
            "quantidade_compra_min": 1,
            "quantidade_compra_max": 30,

             #choque

            "dia":30 ,
            "tipo": "negativo",
            "intensidade": 0.7,
            "duracao": 2,
            "delta": 0.8,
            "prob_choque_diario": 0.025

        },

        # Parâmetros do Mercado
        "mercado": {
            "volatilidade_inicial": 0.1,
            "dividendos_frequencia": 21,            # Distribuição de dividendos a cada 21 dias
            "atualizacao_imoveis_frequencia": 126,    # Atualização dos imóveis a cada 126 dias

        },

        # Parâmetros do OrderBook (pode ser extendido futuramente)
        "order_book": {},

        # Parâmetros para Plotagem
        "plot": {
            "window_volatilidade": 200,
        },
    }

    # --- Inicialização do FII e dos Imóveis ---
    total_cota = sim_params["total_cota"]
    num_agentes = sim_params["num_agentes_pf"] + sim_params["num_agentes_pj"]
    cota_agente = int(total_cota * sim_params["prop_cota_agente"])
    cota_fii = total_cota - cota_agente

    fii = FII(num_cotas=sim_params["fii"]["num_cotas"],
              caixa=sim_params["fii"]["caixa_inicial"],
              params=sim_params["fii"])

    for imovel_param in sim_params["imoveis"]:
        imovel = Imovel(valor=imovel_param["valor"],
                        vacancia=imovel_param["vacancia"],
                        custo_manutencao=imovel_param["custo_manutencao"],
                        params=imovel_param.get("params", None))
        fii.adicionar_imovel(imovel)


    # print(f"Preço inicial da cota: R${fii.preco_cota:,.2f}") # Formatação simplificada

    historia = fii.inicializar_historico(memoria=sim_params["fii"]["memoria"])
    fii.preco_cota = fii.historico_precos[-1]

    historico_diviendos = fii.historico_dividendos

    ########################################################################
    # -------------------- Criação dos Agentes -----------------------------
    ########################################################################

    # --- Criação dos Agentes PF ---
    agentes_pf = []
    num_agentes_pf = sim_params["num_agentes_pf"]

    for i in range(num_agentes_pf):
        cotas_iniciais = (sim_params["agente_pf"]["cotas_iniciais_primeiro"] if i == 0 else sim_params["agente_pf"]["cotas_iniciais_outros"])
        agente = Agente(
            id=i,
            literacia_financeira=gerar_literacia_financeira(minimo=0.2, maximo= 0.7),
                # media=sim_params["agente"]["literacia_media"],
                # desvio=sim_params["agente"]["literacia_std"]
                # ),
            caixa=sim_params["agente_pf"]["caixa_inicial"],
            cotas=cotas_iniciais,
            expectativa_inflacao=sim_params["agente_pf"]["expectativa_inflacao"],
            expectativa_premio=sim_params["agente_pf"]["expectativa_premio"],
            historico_precos=historia, # Passar numpy array
            params=sim_params["agente_pf"].get("params", None)
        )
        agentes_pf.append(agente)

    # Definir vizinhos: pode ser feito em paralelo se a lista `todos_agentes` for estática
    # Mas o custo de serialização pode não compensar para esta etapa de inicialização.

    # --- Criação dos Agentes PJ ---

    agentes_pj = []
    num_agentes_pj = sim_params["num_agentes_pj"]


    for i in range(num_agentes_pj):
        cotas_iniciais = (sim_params["agente_pj"]["cotas_iniciais_primeiro"] if i == 0 else sim_params["agente_pj"]["cotas_iniciais_outros"])
        agente = Agente(
            id=i + num_agentes_pf,
            literacia_financeira=gerar_literacia_financeira(minimo=0.7, maximo= 1),
                # media=sim_params["agente"]["literacia_media"],
                # desvio=sim_params["agente"]["literacia_std"]
                # ),
            caixa=sim_params["agente_pj"]["caixa_inicial"],
            cotas=cotas_iniciais,
            expectativa_inflacao=sim_params["agente_pj"]["expectativa_inflacao"],
            expectativa_premio=sim_params["agente_pj"]["expectativa_premio"],
            historico_precos=historia, # Passar numpy array
            params=sim_params["agente_pj"].get("params", None)
        )
        agentes_pj.append(agente)

    # --- Combinar todos os agentes ---
    agentes = agentes_pf + agentes_pj

    # --- Definir vizinhos: grupos separados ---
    for agente in agentes_pf:
        agente.definir_vizinhos(agentes, num_vizinhos=sim_params["agente_pf"]["num_vizinhos"])

    for agente in agentes_pj:
        agente.definir_vizinhos(agentes_pj, num_vizinhos=int(sim_params["agente_pj"]["num_vizinhos"]))

    print(f"Total de Agentes: {len(agentes)}")



    # --- Criação do Banco Central, Mídia e Mercado ---
    bc = BancoCentral(sim_params["banco_central"])
    midia = Midia(
        dias=sim_params["num_dias"],
        valor_inicial=sim_params["midia"]["valor_inicial"],
        sigma=sim_params["parametros_sentimento"]["sigma_midia"],
        valores_fixos=sim_params["midia"]["valores_fixos"]
    )

    mercado = Mercado(agentes=agentes,
                      imoveis=fii.imoveis,
                      fii=fii,
                      banco_central=bc,
                      midia=midia,
                      params=sim_params["mercado"])

    parametros_sentimento = sim_params["parametros_sentimento"]

    # --- Loop da Simulação ---
    historico_precos_fii = historia
    sentimento_medio_ao_longo_dos_dias = []
    num_dias = sim_params["num_dias"]

    # *** Ponto de otimização principal: Loop diário ***
    # Se os cálculos dos agentes são independentes (como no seu caso para sentimento, expectativas),
    # a paralelização aqui pode ter um impacto significativo.
    # Coloquei o código para `_processar_agente_para_pool` na classe `Mercado` para manter a organização.

    for dia in range(1, num_dias + 1):
        # A chamada a mercado.executar_dia agora usará o Pool.map internamente.
        mercado.executar_dia(parametros_sentimento)

        # O calcular_sentimento_medio já é vetorizado.
        sentimento_medio_ao_longo_dos_dias.append(calcular_sentimento_medio(mercado.agentes))
        historico_precos_fii.append(mercado.fii.preco_cota)

    historico_precos_fii = np.array(historico_precos_fii)
    # np.diff já é eficiente
    log_returns = np.diff(np.log(historico_precos_fii))

    # --- Cálculo da Volatilidade Rolante ---
    window = sim_params["plot"]["window_volatilidade"]
    # np.full_like e np.std já são eficientes
    volatilidade_rolante = np.full_like(log_returns, np.nan)
    # Este loop ainda é Python, mas para o cálculo da volatilidade rolante (uma vez por dia),
    # não é um gargalo tão grande quanto os cálculos de agentes por dia.
    for i in range(window, len(log_returns)):
        volatilidade_rolante[i] = np.std(log_returns[i-window:i]) * (252 ** 0.5)

    if imprimir:

        # --- Exibição Final ---
        print(f"Preço Final da Cota: R${fii.preco_cota:,.2f}")
        print(f"Caixa Final do FII: R${fii.caixa:,.2f}")
        for agente in agentes:
          print(f"Agente {agente.id}: Caixa: R${agente.caixa:,.2f}, Sentimento: {agente.sentimento:.2f}, Riqueza: R${agente.historico_riqueza[-1]:,.2f}")

        # --- Plotagem dos Resultados ---
        dias_array = np.arange(num_dias)
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(dias_array, historico_precos_fii, label="Preço da Cota do FII")
        ax[0].set_title("Evolução do Preço do FII")
        ax[0].set_ylabel("Preço")
        ax[0].legend()
        ax[1].plot(dias_array[1:], volatilidade_rolante, label="Volatilidade Rolante (20 dias)", color="orange")
        ax[1].set_title("Volatilidade Rolante dos Retornos Logarítmicos")
        ax[1].set_ylabel("Volatilidade")
        ax[1].set_xlabel("Dias")
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    return historico_precos_fii, log_returns, volatilidade_rolante, midia, sentimento_medio_ao_longo_dos_dias


