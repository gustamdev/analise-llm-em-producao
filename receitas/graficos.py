"""Paleta e estilo dos gráficos, com cor fixa por tipo de etapa e por modelo."""

import matplotlib.pyplot as plt

# Paleta categórica acessível para daltonismo
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

COR_POR_TIPO = {
    "determinística": PALETA[0],
    "RAG": PALETA[1],
    "LLM + Pydantic": PALETA[2],
    "LLM do agente": PALETA[3],
    "roteamento (LLM barato)": PALETA[4],
}
COR_POR_MODELO = {"gpt-4.1": PALETA[0], "gpt-4.1-mini": PALETA[1], "gpt-4.1-nano": PALETA[2], "roteado": PALETA[4]}

TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c9c8c2",
        "axes.grid": True,
        "grid.color": "#ebeae6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA,
        "font.size": 9,
        "legend.frameon": False,
    })


def barras_horizontais(ax, rotulos, valores, cores, formato="{:.3g}", titulo="", log=False):
    """Barras horizontais com o valor na ponta."""
    posicoes = range(len(rotulos))
    ax.barh(posicoes, valores, color=cores, height=0.55, edgecolor="white", linewidth=2)
    ax.set_yticks(list(posicoes), labels=rotulos)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
    for y, valor in zip(posicoes, valores):
        ax.annotate(formato.format(valor), (valor, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TINTA)
    ax.set_title(titulo, loc="left")
