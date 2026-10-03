"""Exporta runs do LangSmith e agrega custo, latência e volume por etapa."""

from datetime import datetime, timedelta, timezone
from itertools import islice

import pandas as pd
from langsmith import Client

from .config import ARQ_RUNS, PROJETO_LANGSMITH, custo_usd
from .tools import NOMES_TOOLS

TIPO_DA_ETAPA = {
    "check_geladeira": "determinística",
    "verificar_validade": "determinística",
    "buscar_receitas_similares": "RAG",
    "criar_receita": "LLM + Pydantic",
    "roteador": "roteamento (LLM barato)",
    "agente (decisão)": "LLM do agente",
}
ETAPAS_NOMEADAS = [*NOMES_TOOLS, "roteador"]
ETAPA_AGENTE = "agente (decisão)"


def exportar_runs(projetos: list[str] | None = None, horas: int = 24, limite: int = 5000) -> pd.DataFrame:
    """Baixa os runs (todas as etapas de todos os traces) das últimas `horas`."""
    client = Client()
    inicio = datetime.now(timezone.utc) - timedelta(hours=horas)
    linhas = []
    for projeto in projetos or [PROJETO_LANGSMITH]:
        # Paginação pelo SDK (máx. 100 por página na API); total limitado com islice
        for run in islice(client.list_runs(project_name=projeto, start_time=inicio), limite):
            metadados = (run.extra or {}).get("metadata", {})
            linhas.append({
                "projeto": projeto,
                "id": str(run.id),
                "trace_id": str(run.trace_id),
                "dotted_order": run.dotted_order,
                "nome": run.name,
                "run_type": run.run_type,
                "inicio": run.start_time,
                "latencia_s": (run.end_time - run.start_time).total_seconds() if run.end_time else None,
                "modelo": metadados.get("ls_model_name"),
                "tokens_entrada": run.prompt_tokens or 0,
                "tokens_saida": run.completion_tokens or 0,
                "custo_langsmith_usd": float(run.total_cost or 0),
                "erro": run.error is not None,
            })
    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    # Custo pela tabela de preços do config
    df["custo_usd"] = [
        custo_usd(m, ti, to) if rt in ("llm", "embedding") and m else 0.0
        for m, ti, to, rt in zip(df["modelo"], df["tokens_entrada"], df["tokens_saida"], df["run_type"])
    ]
    df["etapa"] = _atribuir_etapas(df)
    return df


def _atribuir_etapas(df: pd.DataFrame) -> list[str | None]:
    """Atribui cada run à tool ou ao roteador mais próximo no `dotted_order`."""
    nome_por_id = dict(zip(df["id"], df["nome"]))
    etapas = []
    for dotted, run_type in zip(df["dotted_order"], df["run_type"]):
        ids_do_caminho = [trecho[-36:] for trecho in dotted.split(".")]
        etapa = next(
            (nome_por_id.get(i) for i in reversed(ids_do_caminho) if nome_por_id.get(i) in ETAPAS_NOMEADAS),
            None,
        )
        if etapa is None and run_type == "llm":
            etapa = ETAPA_AGENTE  # LLM fora de tool/roteador = decisão do agente
        etapas.append(etapa)
    return etapas


def resumo_por_etapa(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa os runs por etapa e soma latência, tokens e custo."""
    # Chamada = run da tool, nó roteador ou cada LLM do agente
    eh_chamada = (df["nome"] == df["etapa"]) | ((df["etapa"] == ETAPA_AGENTE) & (df["run_type"] == "llm"))
    chamadas = df[eh_chamada]
    filhos_pagos = df[df["run_type"].isin(["llm", "embedding"]) & df["etapa"].notna()]

    resumo = chamadas.groupby("etapa").agg(
        chamadas=("id", "count"),
        latencia_p50_s=("latencia_s", "median"),
        latencia_p95_s=("latencia_s", lambda s: s.quantile(0.95)),
    )
    gastos = filhos_pagos.groupby("etapa").agg(
        tokens=("tokens_entrada", "sum"),
        tokens_saida=("tokens_saida", "sum"),
        custo_total_usd=("custo_usd", "sum"),
    )
    resumo = resumo.join(gastos, how="left").fillna({"tokens": 0, "tokens_saida": 0, "custo_total_usd": 0.0})
    resumo["tokens"] = resumo["tokens"] + resumo.pop("tokens_saida")
    resumo["custo_por_chamada_usd"] = resumo["custo_total_usd"] / resumo["chamadas"]
    resumo.insert(0, "tipo", [TIPO_DA_ETAPA.get(e, "?") for e in resumo.index])
    return resumo.sort_values("custo_por_chamada_usd")


def resumo_por_trace(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa os runs por pedido, com latência e custo total."""
    raizes = df[df["dotted_order"].str.count(r"\.") == 0].set_index("trace_id")
    custo = df.groupby("trace_id")["custo_usd"].sum()
    return pd.DataFrame({
        "inicio": raizes["inicio"],
        "latencia_s": raizes["latencia_s"],
        "custo_usd": custo.reindex(raizes.index).fillna(0),
    }).sort_values("inicio")


def salvar_export(df: pd.DataFrame) -> None:
    df.to_csv(ARQ_RUNS, index=False)
