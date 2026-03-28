# Bibliotecas


import os
import warnings
import itertools
import datetime as dt
import pandas as pd
import numpy as np
import random # Usado para random.sample e random.shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy.stats import norm, skew, kurtosis, truncnorm, gaussian_kde, jarque_bera, shapiro # Importar diretamente as funções
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.optimize import curve_fit
from statsmodels.graphics.gofplots import qqplot
from multiprocessing import Pool
import random
import traceback # Importar para printar o traceback em caso de erro
from typing import Iterable, Callable, Optional, List, Dict

# Importar nas classes quando usar função standalone

warnings.filterwarnings("ignore")



# Funções

def moving_block_bootstrap(series, block_size=5, sample_size=60, n_replications=1000, random_state=None):
    """
    Implementa o procedimento Moving Block Bootstrap (MBB) para uma série temporal de retornos.

    Parâmetros:
        series (array-like): Série temporal original de retornos (por exemplo, retornos diários).
        block_size (int): Tamanho de cada bloco (default = 5 dias).
        sample_size (int): Tamanho desejado da amostra bootstrap (default = 252 dias).
        n_replications (int): Número de amostras bootstrap a serem geradas.
        random_state (int): Seed para reprodutibilidade.

    Retorna:
        np.ndarray: Matriz com `n_replications` linhas e `sample_size` colunas (uma réplica por linha).
    """
    if isinstance(series, list):
        series = np.array(series)
    elif isinstance(series, pd.Series):
        series = series.values

    n = len(series)
    n_blocks = n - block_size + 1
    k = int(np.ceil(sample_size / block_size))

    # Geração de blocos sobrepostos
    blocks = np.array([series[i:i+block_size] for i in range(n_blocks)])

    # Inicializa gerador de números aleatórios
    rng = np.random.default_rng(random_state)

    # Armazena as amostras bootstrap
    bootstrap_samples = np.empty((n_replications, sample_size))

    for b in range(n_replications):
        # Sorteia k blocos com reposição
        sampled_blocks = rng.choice(blocks, size=k, replace=True)

        # Concatena blocos e trunca para sample_size
        bootstrap_sample = np.concatenate(sampled_blocks)[:sample_size]
        bootstrap_samples[b] = bootstrap_sample.flatten()

    return bootstrap_samples

def tabela_bootstrap_unica(
    bootstraps: np.ndarray,
    acf_raw_lags: Iterable[int] = (5, 10, 15, 20),
    acf_sq_lags: Iterable[int] = (1, 2),
    acf_abs_lags: Iterable[int] = (1),
    alpha: float = 0.05,
    ddof_var: int = 1,
    fisher_kurtosis: bool = False,  # fisher=False -> curtose "clássica"
    fft: bool = True,
) -> pd.DataFrame:
    """
    Uma única tabela com resumos da distribuição bootstrap (IC percentil):
      - Media, Variancia, Assimetria, Curtose
      - ACF dos retornos, retornos^2 e |retornos| nos lags definidos

    bootstraps: array (B, T) onde cada linha é uma replicação bootstrap.
    """
    X = np.asarray(bootstraps, dtype=float)
    if X.ndim != 2:
        raise ValueError("bootstraps deve ter shape (B, T)")
    B, T = X.shape
    if B < 2:
        raise ValueError("precisa de B>=2")

    q_lo = 100 * (alpha / 2)
    q_hi = 100 * (1 - alpha / 2)

    def summarize(vals: np.ndarray) -> Dict[str, float]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return dict(mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan)
        return dict(
            mean=float(np.mean(vals)),
            sd=float(np.std(vals, ddof=1)),
            lo=float(np.percentile(vals, q_lo)),
            hi=float(np.percentile(vals, q_hi)),
        )

    rows: List[Dict[str, object]] = []

    # ---- Momentos básicos por replicação bootstrap ----
    mean_vals = np.mean(X, axis=1)
    var_vals = np.var(X, axis=1, ddof=ddof_var)
    skew_vals = skew(X, axis=1, bias=False, nan_policy="omit")
    kurt_vals = kurtosis(X, axis=1, fisher=fisher_kurtosis, bias=False, nan_policy="omit")

    for nome, vals in [
        ("Media", mean_vals),
        (f"Variancia(ddof={ddof_var})", var_vals),
        ("Assimetria(skew)", skew_vals),
        (f"Curtose(fisher={fisher_kurtosis})", kurt_vals),
    ]:
        s = summarize(vals)
        rows.append({
            "Momento": nome,
            "Média do bootstrap": s["mean"],
            "DP bootstrap": s["sd"],
            "IC 95% inf": s["lo"],
            "IC 95% sup": s["hi"],
        })

    # ---- ACF por replicação via statsmodels.acf ----
    def add_acf_block(prefix: str, lags: Iterable[int], transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        lags = list(lags)
        if any(l <= 0 for l in lags):
            raise ValueError("todos os lags devem ser >= 1")

        max_lag = max(lags)
        Xtr = X if transform is None else np.apply_along_axis(transform, 1, X)

        # calcula ACF uma vez por replicação e coleta os lags desejados
        for lag in lags:
            vals = np.empty(B, dtype=float)
            for b in range(B):
                acfs = acf(Xtr[b], nlags=max_lag, fft=fft)
                vals[b] = acfs[lag]
            s = summarize(vals)
            rows.append({
                "Momento": f"{prefix}_ACF(lag={lag})",
                "Média do bootstrap": s["mean"],
                "DP bootstrap": s["sd"],
                "IC 95% inf": s["lo"],
                "IC 95% sup": s["hi"],
            })

    add_acf_block("retornos", acf_raw_lags, transform=None)
    add_acf_block("retornos2", acf_sq_lags, transform=lambda x: x**2)
    add_acf_block("abs_retornos", acf_abs_lags, transform=np.abs)

    return pd.DataFrame(rows, columns=[
        "Momento", "Média do bootstrap", "DP bootstrap", "IC 95% inf", "IC 95% sup"
    ])



def calcular_power_law(retornos, max_lag=40, janela=None, plotar=False):
    """
    Ajusta uma Lei de Potência (Power Law) à autocorrelação dos retornos absolutos.

    Parâmetros
    ----------
    retornos : array-like
        Série de retornos (simples ou logarítmicos).
    max_lag : int, opcional
        Número máximo de defasagens (lags) para calcular a ACF. Default = 40.
    janela : int, opcional
        Número de observações finais a usar (ex.: últimos 60 dias).
    plotar : bool, opcional
        Se True, plota a ACF(|r|) e o ajuste da Power Law em escala log-log.

    Retorna
    -------
    dict com:
        a : coeficiente de escala da lei de potência
        b : expoente de decaimento (quanto menor, mais persistência)
        r2 : coeficiente de determinação do ajuste (qualidade do fit)
        lags : vetor de defasagens
        acf_abs : autocorrelação dos retornos absolutos
    """
    # Garantir array e janela
    retornos = np.asarray(retornos)
    if janela is not None and len(retornos) > janela:
        retornos = retornos[-janela:]
    if len(retornos) < 5:
        return {"a": np.nan, "b": np.nan, "r2": np.nan}

    # Função Power Law
    def power(x, a, b):
        return a / np.power(x, b)

    # ACF dos retornos absolutos
    acf_abs = sm.tsa.stattools.acf(np.abs(retornos), nlags=max_lag-1, fft=True)
    lags = np.arange(1, max_lag)
    y = acf_abs[1:]

    # Ajuste via mínimos quadrados
    try:
        popt, _ = curve_fit(power, lags, y, p0=[1, 1], maxfev=5000)
        a, b = popt

        # Calcular R² do ajuste
        y_pred = power(lags, a, b)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    except Exception:
        a, b, r2 = np.nan, np.nan, np.nan

    # Plot opcional
    if plotar:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        plt.scatter(lags, y, label="ACF(|r|)", color="blue")
        plt.plot(lags, power(lags, a, b), color="red", label=f"Ajuste Power Law\nb={b:.3f}, R²={r2:.3f}")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Decaimento da Lei de Potência - ACF(|r|)")
        plt.xlabel("Lag (dias)")
        plt.ylabel("ACF(|r|)")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    return {"a": a, "b": b, "r2": r2, "lags": lags, "acf_abs": y}
    
    
    
    
    
def extrair_powerlaw_bootstrap(
    bootstraps_500: np.ndarray,
    historico_inicial: np.ndarray | None = None,
    max_lag: int = 40,
    b_max: float = 3.0,
) -> pd.DataFrame:
    """
    Para cada réplica bootstrap, calcula b e R² do ajuste Power Law em ACF(|r|).
    Retorna DataFrame com colunas: replica, b, r2.
    """
    B = len(bootstraps_500)
    rows = []

    for i in range(B):
        serie = np.asarray(bootstraps_500[i], dtype=float)

        if historico_inicial is not None:
            serie = np.concatenate([np.asarray(historico_inicial, dtype=float), serie])

        res = calcular_power_law(serie, max_lag=max_lag)
        b = res.get("b", np.nan)
        r2 = res.get("r2", np.nan)

        # Mantém seu filtro original (b < 3) e remove NaNs
        if np.isfinite(b) and (b < b_max) and np.isfinite(r2):
            rows.append({"replica": i, "b": float(b), "r2": float(r2)})

    return pd.DataFrame(rows)





def resumo_distribuicao_com_kde(
    values,
    nome: str,
    alpha: float = 0.05,
    cortes: tuple = (),
    plotar: bool = True,
    bins: int = 25,
    kde_bw: str | float = "scott",
    xlim: tuple | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cria (i) tabela-resumo (1 linha) + (ii) tabela de quantis e (iii) gráfico hist+KDE.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        df_resumo = pd.DataFrame([{"Momento": nome, "n": 0}])
        df_quantis = pd.DataFrame({"quantil": [], "valor": []})
        return df_resumo, df_quantis

    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else np.nan
    med = float(np.median(x))
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    skw = float(skew(x, bias=False)) if n > 2 else np.nan
    krt = float(kurtosis(x, fisher=False, bias=False)) if n > 3 else np.nan
    ci_low = float(np.quantile(x, q_lo))
    ci_high = float(np.quantile(x, q_hi))

    # quantis úteis
    quantis = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}
    df_quantis = pd.DataFrame({
        "quantil": list(quantis.keys()),
        "valor": [float(np.quantile(x, q)) for q in quantis.values()]
    })

    # proporções opcionais
    props = {f"p({nome}>={c})": float(np.mean(x >= c)) for c in cortes} if cortes else {}

    # normalidade (opcional, mas útil pra comentar forma da distribuição)
    jb_stat, jb_p = jarque_bera(x)
    shp_stat = shp_p = np.nan
    if n <= 5000:
        shp_stat, shp_p = shapiro(x)

    df_resumo = pd.DataFrame([{
        "Momento": nome,
        "n": int(n),
        "mean": mean,
        "std": std,
        "median": med,
        "min": xmin,
        "max": xmax,
        "skew": skw,
        "kurtosis": krt,
        f"IC{int((1-alpha)*100)}%_inf": ci_low,
        f"IC{int((1-alpha)*100)}%_sup": ci_high,
        "JB_pvalue": float(jb_p),
        "Shapiro_pvalue": float(shp_p),
        **props
    }])

    if plotar:
        kde = gaussian_kde(x, bw_method=kde_bw) if n >= 5 else None
        xs = np.linspace(xmin, xmax, 500) if xlim is None else np.linspace(xlim[0], xlim[1], 500)
        ys = kde(xs) if kde is not None else None

        plt.figure(figsize=(8, 4.8))
        plt.hist(x, bins=bins, density=True, alpha=0.55, edgecolor="black")
        if ys is not None:
            plt.plot(xs, ys, linewidth=2, label="Densidade (KDE)")

        plt.axvline(med, linestyle="--", linewidth=2, label=f"Mediana={med:.3f}")
        plt.axvline(ci_low, linestyle=":", linewidth=2, label=f"IC inf={ci_low:.3f}")
        plt.axvline(ci_high, linestyle=":", linewidth=2, label=f"IC sup={ci_high:.3f}")

        plt.title(f"Distribuição bootstrap de {nome}")
        plt.xlabel(nome)
        plt.ylabel("Densidade")
        if xlim is not None:
            plt.xlim(*xlim)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_resumo, df_quantis

''' EXEMPLO DE USO

df_power = extrair_powerlaw_bootstrap(
    bootstraps_500=bootstraps_500,
    historico_inicial=historico_inicial,  # coloque None se não quiser concatenar
    max_lag=40,
    b_max=3.0
# ---- Resumo de b ----
df_b_resumo, df_b_quantis = resumo_distribuicao_com_kde(
    df_power["b"],
    nome="b (expoente power law)",
    alpha=0.05,
    cortes=(0.5, 1.0, 1.5, 2.0),   # opcionais, ajuste ao seu texto
    plotar=True,
    bins=25,
    kde_bw="scott"
)

# ---- Resumo de R² ----
df_r2_resumo, df_r2_quantis = resumo_distribuicao_com_kde(
    df_power["r2"],
    nome="R² (qualidade do ajuste)",
    alpha=0.05,
    cortes=(0.6, 0.7, 0.8, 0.9),   # opcional
    plotar=True,
    bins=25,
    kde_bw="scott",
    xlim=(0, 1)                    # R² em [0,1]
)

# ---- Tabela final (2 linhas) para artigo ----
df_artigo = pd.concat([df_b_resumo, df_r2_resumo], ignore_index=True)
df_artigo

'''

def calcular_momentos_uma_serie(
    serie: np.ndarray,
    lags_retornos=(5, 10, 15, 20),
    lags_quad=(1, 2),
    ddof_var=1,
    fisher_kurtosis=False,
    fft=True,
) -> dict:
    """
    Calcula um vetor de momentos (um dicionário) para uma única série de retornos.

    Momentos (compatível com seu df_dados_calib):
      Media, Variancia, Assimetria, Curtose,
      acf_5, acf_10, acf_15, acf_20,
      autocorr_q_1, autocorr_q_2  (ACF de retornos ao quadrado em lags 1 e 2)
    """
    r = np.asarray(serie, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < max(max(lags_retornos, default=0), max(lags_quad, default=0), 2) + 5:
        # tamanho insuficiente
        return {k: np.nan for k in [
            "Media","Variancia","Assimetria","Curtose",
            "acf_5","acf_10","acf_15","acf_20",
            "autocorr_q_1","autocorr_q_2"
        ]}

    out = {}
    out["Media"] = float(np.mean(r))
    out["Variancia"] = float(np.var(r, ddof=ddof_var))
    out["Assimetria"] = float(skew(r, bias=False))
    out["Curtose"] = float(kurtosis(r, fisher=fisher_kurtosis, bias=False))

    # ACF dos retornos em lags selecionados
    max_lag_r = max(lags_retornos)
    acf_r = acf(r, nlags=max_lag_r, fft=fft)
    for L in lags_retornos:
        out[f"acf_{L}"] = float(acf_r[L])

    # ACF dos retornos ao quadrado (autocorr_q_1, autocorr_q_2)
    rq = r**2
    max_lag_q = max(lags_quad)  # aqui usamos (1,2)
    acf_q = acf(rq, nlags=max_lag_q, fft=fft)
    out["autocorr_q_1"] = float(acf_q[1])
    out["autocorr_q_2"] = float(acf_q[2]) if max_lag_q >= 2 else np.nan

    return out


def construir_M_momentos_bootstrap(
    bootstraps: np.ndarray,
    cols_momentos=None,
    lags_retornos=(5, 10, 15, 20),
    ddof_var=1,
    fisher_kurtosis=False,
    fft=True,
) -> pd.DataFrame:
    """
    Constrói M (DataFrame B x k) com momentos calculados em cada réplica bootstrap.
    """
    if cols_momentos is None:
        cols_momentos = [
            "Media","Variancia","Assimetria","Curtose",
            "acf_5","acf_10","acf_15","acf_20",
            "autocorr_q_1","autocorr_q_2"
        ]

    X = np.asarray(bootstraps, dtype=float)
    if X.ndim != 2:
        raise ValueError("bootstraps deve ter shape (B, T)")
    B, T = X.shape

    rows = []
    for b in range(B):
        m = calcular_momentos_uma_serie(
            X[b],
            lags_retornos=lags_retornos,
            ddof_var=ddof_var,
            fisher_kurtosis=fisher_kurtosis,
            fft=fft
        )
        rows.append(m)

    M_df = pd.DataFrame(rows, columns=cols_momentos)
    return M_df



def matriz_variancia_covariancia_dos_momentos(M_df: pd.DataFrame) -> np.ndarray:
    """
    Σ̂ = cov(M) com ddof=1 (B-1 no denominador).
    """
    M = M_df.to_numpy(dtype=float)
    # remove linhas com NaN (se existirem)
    mask = np.all(np.isfinite(M), axis=1)
    M = M[mask]
    if M.shape[0] < 2:
        raise ValueError("Poucas linhas válidas (sem NaN) para estimar covariância.")
    Sigma_hat = np.cov(M, rowvar=False, ddof=1)
    return Sigma_hat

def matriz_pesos_W(Sigma_hat: np.ndarray, regularizacao: float = 1e-8) -> np.ndarray:
    """
    W = (Σ̂ + λI)^(-1) para estabilidade numérica.
    """
    k = Sigma_hat.shape[0]
    Sigma_reg = Sigma_hat + regularizacao * np.eye(k)
    return np.linalg.inv(Sigma_reg)

def calcular_erros_calibracao(
    df_dados_calib: pd.DataFrame,
    momentos_objetivo,
    W,
    cols_momentos=None
) -> pd.DataFrame:
    """
    Para cada linha de df_dados_calib:
      1) extrai vetor de momentos (cols_momentos)
      2) diff = momentos_objetivo - momentos_linha
      3) erro = diff @ W @ diff
    Retorna DataFrame com colunas [a0,b0,c0,beta,p5,erro].

    Params
    ------
    df_dados_calib : DataFrame
        Deve conter as colunas de parâmetros e as colunas de momentos.
    momentos_objetivo : array-like (k,)
        Vetor de momentos alvo, na mesma ordem de cols_momentos.
    W : array-like (k,k)
        Matriz de pesos.
    cols_momentos : list[str] | None
        Lista ordenada de colunas de momentos. Se None, usa o padrão do enunciado.
    """
    if cols_momentos is None:
        cols_momentos = [
            "Media", "Variancia", "Assimetria", "Curtose",
            "acf_5", "acf_10", "acf_15", "acf_20",
            "autocorr_q_1", "autocorr_q_2"
        ]

    cols_params = ["a0", "b0", "c0", "beta", "p5"]

    faltando_params = [c for c in cols_params if c not in df_dados_calib.columns]
    faltando_moms = [c for c in cols_momentos if c not in df_dados_calib.columns]
    if faltando_params:
        raise ValueError(f"Colunas de parâmetros faltando em df_dados_calib: {faltando_params}")
    if faltando_moms:
        raise ValueError(f"Colunas de momentos faltando em df_dados_calib: {faltando_moms}")

    m_obj = np.asarray(momentos_objetivo, dtype=float).reshape(-1)
    W = np.asarray(W, dtype=float)

    k = len(cols_momentos)
    if m_obj.shape[0] != k:
        raise ValueError(f"momentos_objetivo deve ter tamanho {k}, mas tem {m_obj.shape[0]}")
    if W.shape != (k, k):
        raise ValueError(f"W deve ter shape {(k,k)}, mas tem {W.shape}")

    # Matriz de momentos por linha (n, k)
    M = df_dados_calib[cols_momentos].to_numpy(dtype=float)

    # diff por linha: (n, k)
    diff = m_obj[None, :] - M

    # erro por linha: (n,)
    # forma vetorizada: erro_i = diff_i @ W @ diff_i
    erro = np.einsum("ij,jk,ik->i", diff, W, diff)

    out = df_dados_calib[cols_params].copy()
    out["erro"] = erro
    return out
