import random
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import expon

from src.microestrutura import Ordem


def gerar_literacia_financeira(minimo=0.2, maximo=1.0, lambda_=4):
    """
    Gera um valor de literacia financeira usando uma distribuição exponencial
    truncada no intervalo [minimo, maximo].
    A densidade é decrescente: valores mais baixos são mais prováveis.
    """
    while True:
        valor = expon(scale=1/lambda_).rvs()
        if minimo <= valor <= maximo:
            return valor


class Agente:
    def __init__(self, id: int, literacia_financeira: float, caixa: float, cotas: int,
                 expectativa_inflacao: float, expectativa_premio: float,
                 historico_precos: list, params: dict = None) -> None:
        self.id  = id
        self.LF  = literacia_financeira
        self.caixa = caixa
        self.cotas = cotas

        self.params = params if params is not None else {}

        # Probabilidade de negociar — decrescente com a literacia financeira
        self.prob_negociar = np.clip(0.8 - 0.875 * (self.LF - 0.2), 0.1, 0.8)

        self.saldo    = caixa
        self.carteira = {"FII": cotas}

        self.sentimento          = 0
        self.RD                  = 0
        self.percentual_alocacao = 0

        self.expectativa_inflacao = expectativa_inflacao
        self.expectativa_premio   = expectativa_premio

        self.historico_precos  = np.array(historico_precos)
        self.retornos_dia      = []
        self.historico_riqueza = np.array([caixa + cotas * historico_precos[-1]])

        self.historico_sentimentos = []
        self.vizinhos              = []

    # ══════════════════════════════════════════════════════════
    # REDE SOCIAL
    # ══════════════════════════════════════════════════════════

    def definir_vizinhos(self, todos_agentes: list,
                         num_vizinhos: int = 30,
                         epsilon_lf: float = 1.0) -> None:
        """
        Define os vizinhos do agente.

        epsilon_lf : float
            Raio de similaridade de literacia financeira.
            1.0  → rede aleatória (padrão)
            <1.0 → rede cognitiva: vizinhos com |LF_i - LF_j| ≤ epsilon_lf
        """
        outros = [a for a in todos_agentes if a.id != self.id]

        if epsilon_lf >= 1.0:
            candidatos = outros
        else:
            candidatos = [a for a in outros
                          if abs(a.LF - self.LF) <= epsilon_lf]

        if len(candidatos) >= num_vizinhos:
            self.vizinhos = random.sample(candidatos, num_vizinhos)
        else:
            restantes = [a for a in outros if a not in candidatos]
            self.vizinhos = candidatos + random.sample(
                restantes,
                min(num_vizinhos - len(candidatos), len(restantes))
            )

    # ══════════════════════════════════════════════════════════
    # FORMAÇÃO DE EXPECTATIVAS E PREÇO
    # ══════════════════════════════════════════════════════════

    def calcular_preco_esperado(self, literacia_financeira: float,
                                beta: float, dividendos: float) -> float:
        x = literacia_financeira / (np.exp(1) ** beta)
        z = (1 - beta) * (1 - literacia_financeira)
        y = 1 - x - z

        preco_fundamentalista = (
            dividendos * 12 * (1 + self.expectativa_inflacao) / self.expectativa_premio
        )

        if not isinstance(self.historico_precos, np.ndarray):
            self.historico_precos = np.array(self.historico_precos)

        if (len(self.historico_precos) > 0
                and self.historico_precos[-1] > 0
                and preco_fundamentalista > 0):
            retorno_fundamentalista = (
                np.log(preco_fundamentalista) - np.log(self.historico_precos[-1])
            )
        else:
            retorno_fundamentalista = 0.0

        def calcular_sma(precos_historicos, LF):
            if not isinstance(precos_historicos, np.ndarray):
                precos_historicos = np.array(precos_historicos)

            omega        = max(2, int(LF * 252))
            janela_curta = max(2, int(omega / 4))

            sma_short = precos_historicos[-1] if len(precos_historicos) > 0 else 0.0
            sma_long  = precos_historicos[-1] if len(precos_historicos) > 0 else 0.0

            if len(precos_historicos) >= janela_curta:
                sma_short = np.mean(precos_historicos[-janela_curta:])
            if len(precos_historicos) >= omega:
                sma_long = np.mean(precos_historicos[-omega:])

            if sma_long == 0 and sma_short == 0:
                return 0.0, 0.0
            elif sma_long == 0:
                return sma_short, sma_short
            elif sma_short == 0:
                return sma_long, sma_long
            return sma_short, sma_long

        sma_short, sma_long = calcular_sma(self.historico_precos, self.LF)

        retorno_especulador = np.log(sma_short / sma_long) if sma_long > 0 else 0.0

        ruido_std     = self.params.get("ruido_std", 0.1)
        retorno_ruido = np.random.normal(0, ruido_std)

        retorno_expectativa = (
            x * retorno_fundamentalista
            + y * retorno_especulador
            + z * retorno_ruido
        )

        if len(self.historico_precos) > 0 and self.historico_precos[-1] > 0:
            preco_esperado = self.historico_precos[-1] * np.exp(retorno_expectativa)
        else:
            preco_esperado = 0.0

        return preco_esperado

    def atualizar_caixa(self, taxa_selic: float, dividendos: float) -> None:
        taxa_selic_mes = (1 + taxa_selic) ** (1 / 12) - 1
        self.caixa *= (1 + taxa_selic_mes)
        self.caixa += dividendos

    def calcular_I_privada(self, n: int = 5, beta: int = 0,
                           dividendos: float = 0.0) -> float:
        peso_retorno = self.params.get("peso_retorno", 0.8)
        peso_riqueza = self.params.get("peso_riqueza", 0.4)

        preco_esperado = self.calcular_preco_esperado(self.LF, beta, dividendos)

        if not isinstance(self.historico_precos, np.ndarray):
            self.historico_precos = np.array(self.historico_precos)

        preco_atual = self.historico_precos[-1] if len(self.historico_precos) > 0 else 0.0

        if preco_atual > 0 and preco_esperado > 0:
            componente_retorno = np.log(preco_esperado / preco_atual)
        else:
            componente_retorno = 0.0

        if not isinstance(self.historico_riqueza, np.ndarray):
            self.historico_riqueza = np.array(self.historico_riqueza)

        n = int(self.LF * 252)

        if len(self.historico_riqueza) >= n and self.historico_riqueza[-n] != 0:
            variacao_riqueza = (
                (self.historico_riqueza[-1] - self.historico_riqueza[-n])
                / self.historico_riqueza[-n]
            )
            componente_riqueza = variacao_riqueza
        else:
            componente_riqueza = 0.0

        I_privada = (
            peso_retorno * componente_retorno
            + peso_riqueza * componente_riqueza
            + np.random.normal(0, 0.05)
        )
        return I_privada

    def calcular_I_social(self, vizinhos: list) -> float:
        sentimentos_vizinhos = np.array([
            np.mean(v.historico_sentimentos[-3:])
            if len(v.historico_sentimentos) >= 3
            else (np.mean(v.historico_sentimentos)
                  if v.historico_sentimentos else 0.0)
            for v in vizinhos
        ])
        sentimentos_vizinhos = np.nan_to_num(sentimentos_vizinhos)

        if len(sentimentos_vizinhos) == 0:
            return 0.0

        return float(np.mean(sentimentos_vizinhos))

    def calcular_sentimento_risco_alocacao(self, mercado: "Mercado",
                                           vizinhos: list,
                                           parametros: dict) -> None:
        I_privado = self.calcular_I_privada(
            n=5,
            beta=parametros["beta"],
            dividendos=mercado.fii.historico_dividendos[-1],
        )
        I_social = self.calcular_I_social(vizinhos)

        volatilidade_percebida = mercado.volatilidade_historica
        a_i = parametros["a0"] * self.LF
        b_i = parametros["b0"] * (1 - self.LF)
        c_i = parametros["c0"] * (1 - self.LF)

        S_bruto = round(
            a_i * I_privado + b_i * I_social + c_i * mercado.news, 4
        )
        self.sentimento = max(min(S_bruto, 1), -1)
        self.historico_sentimentos.append(self.sentimento)

        self.RD = (self.sentimento + 1) / 2 * volatilidade_percebida
        self.percentual_alocacao = (
            self.RD / volatilidade_percebida if volatilidade_percebida > 0 else 0
        )

    def calcular_expectativa_inflacao(self, banco_central: "BancoCentral",
                                      noticias: float, mercado: "Mercado",
                                      parametros: float) -> None:
        self.expectativa_inflacao = banco_central.expectativa_inflacao * (
            1 - self.sentimento * parametros["peso_sentimento_inflacao"]
        )

    def calcular_expectativa_premio(self, mercado: "Mercado",
                                    parametros: float) -> None:
        self.expectativa_premio = mercado.banco_central.premio_risco * (
            1 - self.sentimento * parametros["peso_sentimento_expectativa"]
        )

    def calcular_estatisticas_retoricas(self) -> Optional[dict]:
        if len(self.historico_precos) < 2:
            return None
        if not isinstance(self.historico_precos, np.ndarray):
            self.historico_precos = np.array(self.historico_precos)

        retornos      = ((self.historico_precos[1:] - self.historico_precos[:-1])
                         / self.historico_precos[:-1])
        media_retorno = np.mean(retornos)
        volatilidade  = np.std(retornos)
        sharpe_ratio  = media_retorno / volatilidade if volatilidade > 0 else 0
        return {
            "media_retorno": media_retorno,
            "volatilidade":  volatilidade,
            "sharpe_ratio":  sharpe_ratio,
        }

    def calcular_retornos_dia(self, preco_atual: float) -> None:
        if len(self.historico_precos) > 0:
            preco_anterior = self.historico_precos[-1]
            if preco_anterior > 0:
                retorno = (preco_atual - preco_anterior) / preco_anterior
                self.retornos_dia.append(retorno)
        self.historico_precos = np.append(self.historico_precos, preco_atual)

    def atualizar_historico(self, preco_fii: float) -> None:
        riqueza_atual = self.caixa + self.carteira.get("FII", 0) * preco_fii
        self.historico_riqueza = np.append(self.historico_riqueza, riqueza_atual)

    def criar_ordem(self, mercado: "Mercado",
                    parametros: float) -> Optional["Ordem"]:
        ativo         = "FII"
        preco_mercado = mercado.fii.preco_cota
        if preco_mercado <= 0:
            return None

        preco_esperado = self.calcular_preco_esperado(
            literacia_financeira=self.LF,
            beta=parametros["beta"],
            dividendos=mercado.fii.historico_dividendos[-1],
        )

        peso_base = parametros.get("peso_preco_esperado", 0.3)

        peso_preco_esperado = (
            peso_base * 2
            if hasattr(self, "choque_ativo") and self.choque_ativo
            else peso_base
        )

        if preco_mercado < preco_esperado:
            qtd_min          = parametros.get("quantidade_compra_min", 1)
            qtd_max          = parametros.get("quantidade_compra_max", 10)
            cotas_desejadas  = random.randint(qtd_min, qtd_max)
            valor_necessario = preco_mercado * cotas_desejadas
            if self.saldo >= valor_necessario:
                preco_limite = (
                    (1 - peso_preco_esperado) * preco_mercado
                    + peso_preco_esperado * preco_esperado
                )
                return Ordem(tipo="compra", agente=self, ativo=ativo,
                             preco_limite=preco_limite,
                             quantidade=cotas_desejadas)

        elif preco_mercado > preco_esperado:
            cotas_possuidas = self.carteira.get(ativo, 0)
            if cotas_possuidas > 0:
                preco_limite  = (
                    (1 - peso_preco_esperado) * preco_mercado
                    + peso_preco_esperado * preco_esperado
                )
                divisor_venda = parametros.get("divisor_quantidade_venda", 5)
                qtd_venda_max = int(cotas_possuidas / divisor_venda) + 1
                return Ordem(tipo="venda", agente=self, ativo=ativo,
                             preco_limite=preco_limite,
                             quantidade=random.randint(1, qtd_venda_max))
        return None

    # ══════════════════════════════════════════════════════════
    # MECANISMO DE CHOQUE
    # ══════════════════════════════════════════════════════════

    def aplicar_choque(self, tipo_choque: str, intensidade: float = 0.2,
                       duracao: int = 5, delta: float = 0.5) -> None:
        """
        Aplica um choque ao agente e inicia dissipação temporal.

        O choque modifica prob_negociar e expectativas de inflação e prêmio.
        Sentimento responde automaticamente via calcular_sentimento_risco_alocacao.
        """
        if not hasattr(self, "_prob_negociar_base"):
            self._prob_negociar_base = self.prob_negociar

        self.choque_ativo = {
            "tipo":                tipo_choque,
            "intensidade_inicial": intensidade,
            "dias_restantes":      duracao,
            "delta":               delta,
            "intensidade_atual":   intensidade,
        }
        self._aplicar_impacto_choque()

    def _aplicar_impacto_choque(self) -> None:
        """
        Modifica expectativas e prob_negociar com base no tipo e intensidade do choque.

        Negativo → eleva expectativas (+intensidade/100) e prob_negociar (pânico)
        Positivo → reduz expectativas (-intensidade/100) e prob_negociar (confiança)
        """
        choque = getattr(self, "choque_ativo", None)
        if not choque:
            return

        intensidade = choque["intensidade_atual"]
        tipo        = choque["tipo"]

        if tipo == "negativo":
            self.expectativa_inflacao *= (1 + intensidade / 100)
            self.expectativa_premio   *= (1 + intensidade / 100)
            self.prob_negociar = np.clip(
                self._prob_negociar_base * (1 + 0.3 * intensidade),
                0.1, 1.0,
            )

        elif tipo == "positivo":
            self.expectativa_inflacao *= (1 - intensidade / 100)
            self.expectativa_premio   *= (1 - intensidade / 100)
            self.prob_negociar = np.clip(
                self._prob_negociar_base * (1 - 0.3 * intensidade),
                0.1, 1.0,
            )

    def atualizar_choque(self) -> None:
        """Atualiza dissipação do choque e restaura valores base ao final."""
        choque = getattr(self, "choque_ativo", None)
        if not choque:
            return

        if choque["dias_restantes"] > 0:
            choque["dias_restantes"]    -= 1
            choque["intensidade_atual"] *= choque["delta"]
            self._aplicar_impacto_choque()
        else:
            self.prob_negociar = self._prob_negociar_base
            self.choque_ativo  = None
