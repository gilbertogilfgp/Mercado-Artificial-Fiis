import copy
import numpy as np
import matplotlib.pyplot as plt

from src.ativos import Imovel, FII
from src.agentes import Agente, gerar_literacia_financeira
from src.mercado import BancoCentral, Midia, Mercado


SIM_PARAMS = {

    "num_dias": 500,
    "total_cota": 100_000,

    # ── FII ───────────────────────────────────────────────────
    "fii": {
        "num_cotas": 100_000,
        "caixa_inicial": 50_000,
        "dividendos_taxa": 0.95,
        "dividendos_caixa_taxa": 0.05,
        "investimento_fracao": 0.50,
        "memoria": True,
    },

    # ── Imóveis ───────────────────────────────────────────────
    "imoveis": [
        {"valor": 1_000_000, "vacancia": 0.1, "custo_manutencao": 200,
         "params": {"aluguel_factor": 0.005, "desvio_normal": 0.01}},
        {"valor": 2_000_000, "vacancia": 0.2, "custo_manutencao": 500,
         "params": {"aluguel_factor": 0.005, "desvio_normal": 0.01}},
    ],

    # ── Agentes PF ────────────────────────────────────────────
    "num_agentes_pf": 600,
    "prop_cota_agente": 0.6,
    "agente_pf": {
        "caixa_inicial": 10_000,
        "cotas_iniciais_primeiro": 100,
        "cotas_iniciais_outros": 100,
        "num_vizinhos": 30,
        "epsilon_lf": 1.0,
        "literacia_media": 0.3,
        "literacia_std": 0.4,
        "expectativa_inflacao": 0.05,
        "expectativa_premio": 0.08,
        "params": {
            "ruido_std": 0.1,
            "peso_retorno": 0.6,
            "peso_riqueza": 0.4,
        },
    },

    # ── Agentes PJ ────────────────────────────────────────────
    "num_agentes_pj": 200,
    "prop_cota_agente_pj": 0.2,
    "agente_pj": {
        "caixa_inicial": 10_000,
        "cotas_iniciais_primeiro": 100,
        "cotas_iniciais_outros": 100,
        "num_vizinhos": 30,
        "epsilon_lf": 1.0,
        "literacia_media": 0.3,
        "literacia_std": 0.4,
        "expectativa_inflacao": 0.05,
        "expectativa_premio": 0.08,
        "params": {
            "ruido_std": 0.05,
            "peso_retorno": 0.3,
            "peso_riqueza": 0.7,
        },
    },

    # ── Banco Central ─────────────────────────────────────────
    "banco_central": {
        "taxa_selic": 0.15,
        "expectativa_inflacao": 0.07,
        "premio_risco": 0.08,
    },

    # ── Mídia ─────────────────────────────────────────────────
    "midia": {
        "valor_inicial": 0,
        "sigma": 0.8,        # controla a volatilidade das notícias
        "valores_fixos": {},
    },

    # ── Parâmetros de sentimento ──────────────────────────────
    # a0, b0, c0, beta, peso_preco_esperado são injetados
    # automaticamente via parametros_sistema
    "parametros_sentimento": {
        "ruido_std": 0.1,
        "piso_prob_negociar": 0.1,
        "peso_sentimento_inflacao": 0.4,
        "peso_sentimento_expectativa": 0.4,
        "quantidade_compra_min": 1,
        "quantidade_compra_max": 30,
        "window_volatilidade": 20,
    },

    # ── Choques aleatórios ────────────────────────────────────
    "prob_choque_diario": 0.025,

    # ── Choques programados ───────────────────────────────────
    # Lista de choques a aplicar em dias específicos.
    # Deixe vazia [] para nenhum choque programado.
    #
    # categoria "noticia" → aplica_choque() em cada agente (dissipa)
    #   campos: dia, tipo, intensidade, duracao, delta
    #
    # categoria "macro" → modifica BancoCentral (permanente)
    #   campos: dia, inflacao, premio
    #
    # categoria "micro" → modifica imóveis do FII (permanente)
    #   campos: dia, vac (%), custo (%)
    #
    "choques": [],

    # ── Mercado ───────────────────────────────────────────────
    "mercado": {
        "volatilidade_inicial": 0.1,
        "dividendos_frequencia": 21,
        "atualizacao_imoveis_frequencia": 126,
    },

    # ── Order Book ────────────────────────────────────────────
    "order_book": {},

    # ── Plot ──────────────────────────────────────────────────
    "plot": {
        "window_volatilidade": 20,
    },
}


# ══════════════════════════════════════════════════════════════
# FUNÇÃO AUXILIAR
# ══════════════════════════════════════════════════════════════

def calcular_sentimento_medio(agentes):
    return float(np.mean([agente.sentimento for agente in agentes]))

# ══════════════════════════════════════════════════════════════
# GRÁFICOS
# ══════════════════════════════════════════════════════════════

def plotar_resultado(resultado, params, num_dias, choques=None):
    """
    Gera o gráfico de 4 painéis a partir do resultado do simulador.

    Parâmetros
    ----------
    resultado : tuple
        Retorno de simular_mercado_e_plotar — (precos, retornos, vol, midia, sentimento)
    params : dict
        SIM_PARAMS usado na simulação — para leitura das cores e choques
    num_dias : int
        Número de dias simulados
    choques : list, opcional
        Lista de choques programados para marcação no gráfico.
        Se None, usa params["choques"].
    """
    import matplotlib.pyplot as plt
    import numpy as np

    precos      = resultado[0]
    retornos    = resultado[1]
    volatilidade = resultado[2]
    sentimento  = resultado[4]

    choques = choques if choques is not None else params.get("choques", [])

    NUM_DIAS    = num_dias
    n_historico = len(precos) - NUM_DIAS

    precos_hist   = np.array(precos[:n_historico])
    precos_sim    = np.array(precos[n_historico:])
    dias_hist     = np.arange(-n_historico + 1, 1)
    dias_sim      = np.arange(1, NUM_DIAS + 1)

    retornos_hist = np.array(retornos)[:-NUM_DIAS]
    retornos_sim  = np.array(retornos)[-NUM_DIAS:]
    dias_ret_hist = np.arange(-len(retornos_hist) + 1, 1)
    dias_ret_sim  = np.arange(1, NUM_DIAS + 1)

    sentimento_arr = np.array(sentimento)
    dias_sent      = np.arange(1, len(sentimento_arr) + 1)

    vol_completa  = np.array(volatilidade)
    vol_hist      = vol_completa[:n_historico - 1]
    vol_sim       = vol_completa[n_historico - 1:]
    dias_vol_hist = np.arange(-n_historico + 2, 1)
    dias_vol_sim  = np.arange(1, len(vol_sim) + 1)

    COR_HIST = "#9EAFC2"
    COR_SIM  = "#1a1a2e"
    x_min = int(dias_hist[0]) - 1
    x_max = NUM_DIAS + 1

    CORES_CHOQUE = {
        "noticia_negativo": "#d62728",
        "noticia_positivo": "#2ca02c",
        "macro":            "#ff7f0e",
        "micro":            "#9467bd",
    }

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)
    fig.suptitle(
        "Mercado Artificial de FIIs\n"
        r"$(a_0,\ b_0,\ c_0,\ \beta,\ \omega)$ = θ*",
        fontsize=12, fontweight="bold"
    )

    # Painel 1 — Preço
    axes[0].plot(dias_hist, precos_hist, color=COR_HIST, lw=1.4, label="Histórico", zorder=2)
    axes[0].fill_between(dias_hist, precos_hist, precos_hist.min(), color=COR_HIST, alpha=0.10)
    axes[0].plot(dias_sim, precos_sim, color=COR_SIM, lw=1.8, label="Simulado", zorder=3)
    axes[0].fill_between(dias_sim, precos_sim, precos_sim.min(), color=COR_SIM, alpha=0.07)
    axes[0].axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.4)
    axes[0].set_ylabel("Preço (R$)", fontsize=10)
    axes[0].set_xlim(x_min, x_max)
    axes[0].grid(True, alpha=0.25, ls=":")

    # Painel 2 — Retornos
    axes[1].bar(dias_ret_hist, retornos_hist,
                color=[COR_HIST if r >= 0 else "#d62728" for r in retornos_hist],
                width=0.8, alpha=0.50)
    axes[1].bar(dias_ret_sim, retornos_sim,
                color=["#2ca02c" if r >= 0 else "#d62728" for r in retornos_sim],
                width=0.8, alpha=0.75)
    axes[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[1].axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.4)
    axes[1].set_ylabel("Retorno log-diário", fontsize=10)
    axes[1].set_xlim(x_min, x_max)
    axes[1].grid(True, alpha=0.25, ls=":")

    # Painel 3 — Sentimento
    axes[2].plot(dias_sent, sentimento_arr, color="#ff7f0e", lw=1.6, label="Sentimento médio")
    axes[2].axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    axes[2].fill_between(dias_sent, sentimento_arr, 0,
                         where=(sentimento_arr >= 0), color="#2ca02c", alpha=0.15, label="Positivo")
    axes[2].fill_between(dias_sent, sentimento_arr, 0,
                         where=(sentimento_arr < 0), color="#d62728", alpha=0.15, label="Negativo")
    axes[2].axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.4)
    axes[2].set_ylabel("Sentimento médio", fontsize=10)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_xlim(x_min, x_max)
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(True, alpha=0.25, ls=":")

    # Painel 4 — Volatilidade
    axes[3].plot(dias_vol_hist, vol_hist, color=COR_HIST, lw=1.4, label="Histórico", zorder=2)
    axes[3].plot(dias_vol_sim, vol_sim, color="#e377c2", lw=1.8, label="Simulado", zorder=3)
    axes[3].fill_between(dias_vol_sim, vol_sim, 0, color="#e377c2", alpha=0.10)
    axes[3].axvline(0.5, color="black", lw=1.0, ls="--", alpha=0.4)
    axes[3].set_ylabel("Volatilidade\nanualizada", fontsize=10)
    axes[3].set_xlim(x_min, x_max)
    axes[3].set_xlabel("Dia (0 = início da simulação)", fontsize=10)
    axes[3].legend(fontsize=8, loc="upper left")
    axes[3].grid(True, alpha=0.25, ls=":")

    # Marcação dos choques
    for choque in choques:
        dia = choque["dia"]
        cat = choque["categoria"]
        if cat == "noticia":
            tipo = choque.get("tipo", "negativo")
            cor  = CORES_CHOQUE[f"noticia_{tipo}"]
            lbl  = f"Notícia {tipo} (dia {dia})"
        elif cat == "macro":
            cor = CORES_CHOQUE["macro"]
            lbl = f"Macro (dia {dia})"
        else:
            cor = CORES_CHOQUE["micro"]
            lbl = f"Micro FII (dia {dia})"
        for ax in axes:
            ax.axvline(dia, color=cor, lw=1.5, ls=":", alpha=0.85,
                       label=lbl if ax == axes[0] else "")

    axes[0].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("simulacao_resultado.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()




# ══════════════════════════════════════════════════════════════
# SIMULADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════

def simular_mercado_e_plotar(
    parametros_sistema,
    num_dias=None,
    sim_params=None,
    imprimir=False,
):
    """
    Simula o mercado artificial de FIIs.

    Parâmetros
    ----------
    parametros_sistema : list
        Vetor calibrado [a0, b0, c0, beta, peso_preco_esperado].
    num_dias : int, opcional
        Sobrescreve SIM_PARAMS["num_dias"] se fornecido.
    sim_params : dict, opcional
        Dicionário de parâmetros. Se None, usa SIM_PARAMS global.
        Para experimentos passe copy.deepcopy(SIM_PARAMS) modificado.
    imprimir : bool
        Se True, exibe estatísticas e gráficos ao final.

    Retorna
    -------
    tuple
        (historico_precos, log_returns, volatilidade_rolante,
         midia, sentimento_medio_ao_longo_dos_dias)
    """

    # ── Configuração ──────────────────────────────────────────
    params = copy.deepcopy(sim_params if sim_params is not None else SIM_PARAMS)

    if num_dias is not None:
        params["num_dias"] = num_dias

    # Injetar parâmetros calibráveis
    params["parametros_sentimento"].update({
        "a0":                  parametros_sistema[0],
        "b0":                  parametros_sistema[1],
        "c0":                  parametros_sistema[2],
        "beta":                parametros_sistema[3],
        "peso_preco_esperado": parametros_sistema[4],
    })

    # Injetar prob_choque_diario
    params["parametros_sentimento"]["prob_choque_diario"] = params.get(
        "prob_choque_diario", 0.025
    )

    # ── FII e Imóveis ─────────────────────────────────────────
    fii = FII(
        num_cotas=params["fii"]["num_cotas"],
        caixa=params["fii"]["caixa_inicial"],
        params=params["fii"],
    )

    for ip in params["imoveis"]:
        fii.adicionar_imovel(Imovel(
            valor=ip["valor"],
            vacancia=ip["vacancia"],
            custo_manutencao=ip["custo_manutencao"],
            params=ip.get("params", None),
        ))

    historia       = fii.inicializar_historico(memoria=params["fii"]["memoria"])
    fii.preco_cota = fii.historico_precos[-1]

    # ── Agentes PF ────────────────────────────────────────────
    agentes_pf     = []
    num_agentes_pf = params["num_agentes_pf"]

    for i in range(num_agentes_pf):
        cotas = (params["agente_pf"]["cotas_iniciais_primeiro"]
                 if i == 0
                 else params["agente_pf"]["cotas_iniciais_outros"])
        agentes_pf.append(Agente(
            id=i,
            literacia_financeira=gerar_literacia_financeira(minimo=0.2, maximo=0.7),
            caixa=params["agente_pf"]["caixa_inicial"],
            cotas=cotas,
            expectativa_inflacao=params["agente_pf"]["expectativa_inflacao"],
            expectativa_premio=params["agente_pf"]["expectativa_premio"],
            historico_precos=historia,
            params=params["agente_pf"].get("params", None),
        ))

    # ── Agentes PJ ────────────────────────────────────────────
    agentes_pj     = []
    num_agentes_pj = params["num_agentes_pj"]

    for i in range(num_agentes_pj):
        cotas = (params["agente_pj"]["cotas_iniciais_primeiro"]
                 if i == 0
                 else params["agente_pj"]["cotas_iniciais_outros"])
        agentes_pj.append(Agente(
            id=i + num_agentes_pf,
            literacia_financeira=gerar_literacia_financeira(minimo=0.7, maximo=1.0),
            caixa=params["agente_pj"]["caixa_inicial"],
            cotas=cotas,
            expectativa_inflacao=params["agente_pj"]["expectativa_inflacao"],
            expectativa_premio=params["agente_pj"]["expectativa_premio"],
            historico_precos=historia,
            params=params["agente_pj"].get("params", None),
        ))

    # ── Combinar e definir vizinhos ───────────────────────────
    agentes = agentes_pf + agentes_pj

    for agente in agentes_pf:
        agente.definir_vizinhos(
            agentes,
            num_vizinhos=params["agente_pf"]["num_vizinhos"],
            epsilon_lf=params["agente_pf"].get("epsilon_lf", 1.0),
        )

    for agente in agentes_pj:
        agente.definir_vizinhos(
            agentes_pj,
            num_vizinhos=int(params["agente_pj"]["num_vizinhos"]),
            epsilon_lf=params["agente_pj"].get("epsilon_lf", 1.0),
        )

    print(f"Total de Agentes: {len(agentes)}")

    # ── Banco Central, Mídia e Mercado ────────────────────────
    bc = BancoCentral(params["banco_central"])

    midia = Midia(
        dias=params["num_dias"],
        valor_inicial=params["midia"]["valor_inicial"],
        sigma=params["midia"]["sigma"],
        valores_fixos=params["midia"]["valores_fixos"],
    )

    mercado = Mercado(
        agentes=agentes,
        imoveis=fii.imoveis,
        fii=fii,
        banco_central=bc,
        midia=midia,
        params=params["mercado"],
    )

    parametros_sentimento = params["parametros_sentimento"]
    choques               = params.get("choques", [])
    num_dias_sim          = params["num_dias"]

    # ── Loop da simulação ─────────────────────────────────────
    historico_precos_fii               = list(historia)
    sentimento_medio_ao_longo_dos_dias = []

    for dia in range(1, num_dias_sim + 1):

        mercado.executar_dia(parametros_sentimento)

        # ── Choques programados ───────────────────────────────
        for choque in choques:
            if choque.get("dia") != dia:
                continue

            categoria = choque.get("categoria", "noticia")

            if categoria == "noticia":
                for ag in mercado.agentes:
                    ag.aplicar_choque(
                        tipo_choque  = choque.get("tipo", "negativo"),
                        intensidade  = choque.get("intensidade", 0.5),
                        duracao      = choque.get("duracao", 2),
                        delta        = choque.get("delta", 0.8),
                    )
                if imprimir:
                    print(f"  💥 Dia {dia}: choque notícia "
                          f"{choque.get('tipo')} "
                          f"(int={choque.get('intensidade')}, "
                          f"dur={choque.get('duracao')}, "
                          f"δ={choque.get('delta')})")

            elif categoria == "macro":
                if "inflacao" in choque:
                    mercado.banco_central.expectativa_inflacao = choque["inflacao"]
                if "premio" in choque:
                    mercado.banco_central.premio_risco = choque["premio"]
                if imprimir:
                    print(f"  📊 Dia {dia}: choque macro "
                          f"(inflação={choque.get('inflacao')}, "
                          f"prêmio={choque.get('premio')})")

            elif categoria == "micro":
                for im in mercado.fii.imoveis:
                    if "vac" in choque:
                        im.vacancia *= (1 + choque["vac"] / 100)
                    if "custo" in choque:
                        im.custo_manutencao *= (1 + choque["custo"] / 100)
                if imprimir:
                    print(f"  🏠 Dia {dia}: choque micro FII "
                          f"(vac={choque.get('vac')}%, "
                          f"custo={choque.get('custo')}%)")

        sentimento_medio_ao_longo_dos_dias.append(
            calcular_sentimento_medio(mercado.agentes)
        )
        historico_precos_fii.append(mercado.fii.preco_cota)

    # ── Pós-processamento ─────────────────────────────────────
    historico_precos_fii = np.array(historico_precos_fii)
    log_returns          = np.diff(np.log(historico_precos_fii))

    window               = params["plot"]["window_volatilidade"]
    volatilidade_rolante = np.full_like(log_returns, np.nan)

    for i in range(window, len(log_returns)):
        volatilidade_rolante[i] = (
            np.std(log_returns[i - window:i]) * (252 ** 0.5)
        )

    # ── Plot ──────────────────────────────────────────────────
    if imprimir:
            plotar_resultado(
                resultado=(historico_precos_fii, log_returns, volatilidade_rolante,
                           midia, sentimento_medio_ao_longo_dos_dias),
                params=params,
                num_dias=num_dias_sim,
            )
    

    return (
        historico_precos_fii,
        log_returns,
        volatilidade_rolante,
        midia,
        sentimento_medio_ao_longo_dos_dias,
    )
