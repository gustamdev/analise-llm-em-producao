"""Dashboard de custo, latência e qualidade."""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from receitas.config import ARQ_RUNS  # noqa: E402
from receitas.custos import resumo_por_etapa, resumo_por_trace  # noqa: E402
from receitas.graficos import COR_POR_MODELO, COR_POR_TIPO  # noqa: E402
from receitas.tendencia import analisar_tendencia, carregar_historico  # noqa: E402

st.set_page_config(page_title="Assistente de receitas · observabilidade", layout="wide")
st.title("Assistente de receitas · custo, latência e qualidade")

ESCALA_TIPO = alt.Scale(domain=list(COR_POR_TIPO), range=list(COR_POR_TIPO.values()))

# Operação
if not ARQ_RUNS.exists():
    st.info("Sem dados de runs ainda. Rode o notebook 04 para gerar data/runs_export.csv.")
else:
    runs = pd.read_csv(ARQ_RUNS, parse_dates=["inicio"])
    projetos = sorted(runs["projeto"].unique())
    escolhidos = st.multiselect("Projeto do LangSmith", projetos, default=projetos)
    runs = runs[runs["projeto"].isin(escolhidos)]

    if runs.empty:
        st.warning("Nenhum run para os filtros escolhidos.")
    else:
        por_trace = resumo_por_trace(runs)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pedidos (traces)", f"{len(por_trace)}")
        k2.metric("Custo total", f"US$ {por_trace['custo_usd'].sum():.4f}")
        k3.metric("Custo médio por pedido", f"US$ {por_trace['custo_usd'].mean():.5f}")
        k4.metric("Latência p50 por pedido", f"{por_trace['latencia_s'].median():.1f} s")

        resumo = resumo_por_etapa(runs).reset_index()
        st.subheader("Contraste por etapa: determinística × RAG × LLM")

        def grafico(campo: str, titulo: str, log: bool = False) -> alt.Chart:
            escala_x = alt.Scale(type="log") if log else alt.Scale(zero=True)
            return (
                alt.Chart(resumo, title=titulo)
                .mark_bar(height=16, cornerRadiusEnd=4)
                .encode(
                    x=alt.X(f"{campo}:Q", title=None, scale=escala_x),
                    y=alt.Y("etapa:N", title=None, sort=alt.SortField("latencia_p50_s")),
                    color=alt.Color("tipo:N", scale=ESCALA_TIPO, legend=alt.Legend(title="tipo", orient="bottom")),
                    tooltip=[
                        "etapa", "tipo",
                        alt.Tooltip("chamadas:Q", title="chamadas"),
                        alt.Tooltip("latencia_p50_s:Q", title="latência p50 (s)", format=".4f"),
                        alt.Tooltip("latencia_p95_s:Q", title="latência p95 (s)", format=".4f"),
                        alt.Tooltip("custo_por_chamada_usd:Q", title="US$ por chamada", format=".6f"),
                        alt.Tooltip("custo_total_usd:Q", title="US$ total", format=".5f"),
                    ],
                )
                .properties(height=230)
            )

        c1, c2, c3 = st.columns(3)
        c1.altair_chart(grafico("latencia_p50_s", "Latência p50 por chamada (s, log)", log=True), width="stretch")
        c2.altair_chart(grafico("custo_por_chamada_usd", "Custo médio por chamada (US$)"), width="stretch")
        c3.altair_chart(grafico("chamadas", "Volume de chamadas"), width="stretch")

        with st.expander("Tabela por etapa"):
            st.dataframe(resumo, width="stretch")

# Qualidade
st.subheader("Qualidade ao longo das execuções (LLM-as-judge)")
historico = carregar_historico()
if historico.empty:
    st.info("Sem histórico de avaliações. Rode os notebooks 03 e/ou 05.")
else:
    por_execucao = (
        historico.groupby(["registrado_em", "rotulo"], as_index=False)
        .agg(juiz_qualidade=("juiz_qualidade", "mean"), tools_exatas=("tools_exatas", "mean"),
             custo_usd=("custo_usd", "mean"))
        .sort_values("registrado_em")
    )
    tendencia = analisar_tendencia(por_execucao["juiz_qualidade"].tolist())
    por_execucao["media_movel"] = tendencia.get("media_movel", [None] * len(por_execucao))

    t1, t2, t3 = st.columns(3)
    t1.metric("Veredito", tendencia["veredito"])
    t2.metric("Δ janela recente × anterior", f"{tendencia.get('delta', 0):+.3f}")
    t3.metric("Contador Mann-Kendall (S)", f"{tendencia.get('mann_kendall_S', 0):+d}")

    base = alt.Chart(por_execucao).encode(x=alt.X("registrado_em:T", title=None))
    linha = base.mark_line(strokeWidth=2, color=COR_POR_MODELO["gpt-4.1"]).encode(
        y=alt.Y("juiz_qualidade:Q", title="nota média do juiz", scale=alt.Scale(domain=[0, 1])))
    pontos = base.mark_point(size=70, filled=True, color=COR_POR_MODELO["gpt-4.1"]).encode(
        y="juiz_qualidade:Q",
        tooltip=["rotulo", alt.Tooltip("juiz_qualidade:Q", format=".3f"), alt.Tooltip("tools_exatas:Q", format=".2f"),
                 alt.Tooltip("custo_usd:Q", title="US$ por pedido", format=".5f"), "registrado_em:T"])
    media = base.mark_line(strokeDash=[5, 4], strokeWidth=2, color="#52514e").encode(y="media_movel:Q")
    st.altair_chart((linha + media + pontos).properties(height=280), width="stretch")

    st.markdown("**Por modelo × complexidade** (todas as execuções registradas)")
    tabela = historico.pivot_table(index="rotulo", columns="complexidade", values="juiz_qualidade", aggfunc="mean")
    st.dataframe(tabela.round(3), width="stretch")
