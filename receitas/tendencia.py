"""Histórico de avaliações e análise de tendência (janelas + Mann-Kendall)."""

from datetime import datetime

import pandas as pd

from .config import ARQ_HISTORICO

COLUNAS_METRICAS = [
    "feedback.juiz_qualidade", "feedback.tools_exatas", "feedback.tools_f1",
    "feedback.schema_valido", "feedback.aproveitamento_prioritarios", "feedback.roteamento_correto",
]


def resultados_para_df(resultados) -> pd.DataFrame:
    """Converte o resultado do `evaluate()` em tabela, uma linha por execução."""
    df = resultados.to_pandas()
    colunas = {
        "reference.id_cenario": "id_cenario",
        "reference.complexidade": "complexidade",
        "reference.historico_status": "historico_status",
        "outputs.modelo": "modelo",
        "outputs.complexidade_roteada": "complexidade_roteada",
        "outputs.custo_usd": "custo_usd",
        "outputs.latencia_s": "latencia_s",
        "outputs.tokens_entrada": "tokens_entrada",
        "outputs.tokens_saida": "tokens_saida",
        "outputs.tools_chamadas": "tools_chamadas",
    }
    tabela = df[[c for c in colunas if c in df.columns]].rename(columns=colunas)
    for metrica in COLUNAS_METRICAS:
        if metrica in df.columns:
            tabela[metrica.removeprefix("feedback.")] = pd.to_numeric(df[metrica], errors="coerce")
    tabela["experimento"] = resultados.experiment_name
    return tabela


def registrar_no_historico(tabela: pd.DataFrame, rotulo: str) -> None:
    """Acrescenta as linhas de um experimento ao CSV de histórico."""
    tabela = tabela.assign(rotulo=rotulo, registrado_em=datetime.now().isoformat(timespec="seconds"))
    tabela = tabela.assign(tools_chamadas=tabela["tools_chamadas"].astype(str))
    cabecalho = not ARQ_HISTORICO.exists()
    tabela.to_csv(ARQ_HISTORICO, mode="a", header=cabecalho, index=False)


def carregar_historico() -> pd.DataFrame:
    if not ARQ_HISTORICO.exists():
        return pd.DataFrame()
    return pd.read_csv(ARQ_HISTORICO, parse_dates=["registrado_em"])


def contador_mann_kendall(notas: list[float]) -> int:
    """Soma +1 para cada par em que a nota subiu e -1 quando caiu."""
    s = 0
    for i in range(len(notas)):
        for j in range(i + 1, len(notas)):
            s += (notas[j] > notas[i]) - (notas[j] < notas[i])
    return s


def analisar_tendencia(notas: list[float], janela: int = 3, margem: float = 0.03) -> dict:
    """Classifica a série de notas como melhorando, piorando ou estável."""
    serie = pd.Series(notas, dtype=float).dropna().reset_index(drop=True)
    if len(serie) < 2:
        return {"veredito": "dados insuficientes", "n": len(serie)}

    janela = min(janela, len(serie) // 2)  # poucas execuções: divide a série ao meio
    recente = serie.iloc[-janela:]
    anterior = serie.iloc[-2 * janela : -janela]
    delta = recente.mean() - anterior.mean()

    s = contador_mann_kendall(serie.tolist())
    pares = len(serie) * (len(serie) - 1) / 2
    tau = s / pares  # -1 (sempre caindo) ... +1 (sempre subindo)

    if delta > margem and s > 0:
        veredito = "melhorando"
    elif delta < -margem and s < 0:
        veredito = "piorando"
    else:
        veredito = "estável"
    return {
        "veredito": veredito,
        "n": len(serie),
        "media_janela_anterior": round(float(anterior.mean()), 3),
        "media_janela_recente": round(float(recente.mean()), 3),
        "delta": round(float(delta), 3),
        "mann_kendall_S": s,
        "tau": round(tau, 2),
        "media_movel": serie.rolling(janela, min_periods=1).mean().round(3).tolist(),
    }
