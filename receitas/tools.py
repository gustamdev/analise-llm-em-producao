"""As 4 tools do agente."""

from typing import Annotated

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, ValidationError

from .geladeira import classificar_validade, formatar_item
from .llm import obter_llm
from .rag import buscar_receitas
from .schemas import Receita


@tool
def check_geladeira(
    geladeira: Annotated[dict, InjectedState("geladeira")],
    categoria: str | None = None,
) -> str:
    """Lista TODOS os ingredientes da geladeira do usuário, com quantidade e validade.

    Use quando precisar conhecer o inventário completo (ex.: "o que posso fazer com o que
    tenho?"). `categoria` é opcional e filtra por categoria (ex.: "hortaliças", "carnes").
    """
    itens = geladeira["itens"]
    if categoria:
        itens = [i for i in itens if categoria.lower() in i["categoria"].lower()]

    linhas = [f"Hoje é {geladeira['data_referencia']}. {len(itens)} itens na geladeira:"]
    categoria_atual = None
    for item in itens:
        if item["categoria"] != categoria_atual:
            categoria_atual = item["categoria"]
            linhas.append(f"[{categoria_atual}]")
        linhas.append(formatar_item(item))
    return "\n".join(linhas)


@tool
def verificar_validade(
    geladeira: Annotated[dict, InjectedState("geladeira")],
    dias_limite: int = 3,
) -> str:
    """Retorna só os itens VENCIDOS (devem ser descartados) e os que VENCEM em até
    `dias_limite` dias (devem ser usados primeiro), com quantidades.

    Use quando o usuário quer evitar desperdício / aproveitar o que está para vencer.
    Se o pedido é SÓ sobre esses itens, esta tool basta — não precisa de check_geladeira.
    """
    resultado = classificar_validade(geladeira, dias_limite)
    linhas = [f"Hoje é {geladeira['data_referencia']}."]
    linhas.append("VENCIDOS (descartar, NÃO usar por segurança alimentar):")
    linhas += [formatar_item(i) for i in resultado["vencidos"]] or ["- nenhum"]
    linhas.append(f"VENCENDO em até {dias_limite} dias (usar com prioridade):")
    linhas += [
        f"{formatar_item(i)} -> vence em {i['dias_para_vencer']} dia(s)" for i in resultado["vencendo"]
    ] or ["- nenhum"]
    return "\n".join(linhas)


class _EntradaBusca(BaseModel):
    consulta: str = Field(description="Prato, estilo ou ingredientes principais, de preferência em inglês")
    k: int = Field(default=3, ge=1, le=8, description="Quantas receitas recuperar (mais = mais tokens)")


@tool(args_schema=_EntradaBusca, response_format="content_and_artifact")
def buscar_receitas_similares(consulta: str, k: int = 3) -> tuple[str, dict]:
    """Busca receitas REAIS parecidas numa base de ~2 mil receitas (Epicurious, em inglês).

    Use SÓ quando o usuário pede inspiração num prato/estilo específico ou quer algo
    baseado em receitas conhecidas. Para pedidos simples, NÃO use (custa tokens e tempo).
    """
    docs, uso = buscar_receitas(consulta, k=k)
    blocos = [
        f"### {d.metadata['titulo']}\n{d.page_content.split(chr(10), 1)[1]}\n"
        f"Preparo (resumo): {d.metadata['instrucoes'][:400]}"
        for d in docs
    ]
    # Artifact com o custo do embedding (fora do prompt)
    return "\n\n".join(blocos) or "Nenhuma receita encontrada.", uso


PROMPT_CHEF = """Você é um chef que cria receitas caseiras, em português do Brasil.
Regras:
1. Use SOMENTE os ingredientes informados + básicos de despensa (sal, pimenta-do-reino,
   óleo, azeite, açúcar, farinha de trigo, água). Não invente outros ingredientes.
2. Aproveite o máximo possível dos ingredientes prioritários (vencem logo) e liste-os
   em `ingredientes_prioritarios_usados`.
3. Nunca use itens vencidos.
4. Se houver receitas de referência, inspire-se nelas, adaptando aos ingredientes disponíveis.
5. Quantidades realistas, passos curtos e em ordem lógica."""


@tool(response_format="content_and_artifact")
def criar_receita(
    modelo: Annotated[str, InjectedState("modelo")],
    pedido: str,
    ingredientes: list[str],
    ingredientes_prioritarios: list[str] | None = None,
    receitas_referencia: str | None = None,
) -> tuple[str, dict]:
    """Gera UMA receita estruturada e validada (nome, ingredientes, modo de preparo,
    tempo, porções). Use sempre que for entregar uma receita ao usuário.

    - pedido: o que o usuário quer (ex.: "jantar rápido", "algo estilo risoto")
    - ingredientes: itens disponíveis na geladeira que podem ser usados
    - ingredientes_prioritarios: itens que vencem logo e devem ser aproveitados
    - receitas_referencia: resumo de receitas recuperadas na busca, se houver
    """
    contexto = [f"Pedido: {pedido}", "Ingredientes disponíveis: " + ", ".join(ingredientes)]
    if ingredientes_prioritarios:
        contexto.append("Prioritários (vencem logo): " + ", ".join(ingredientes_prioritarios))
    if receitas_referencia:
        contexto.append(f"Receitas de referência:\n{receitas_referencia}")

    # Gera a receita no schema `Receita`, validada pelo Pydantic
    gerador = obter_llm(modelo).with_structured_output(Receita, method="function_calling")
    try:
        receita: Receita = gerador.invoke(
            [SystemMessage(PROMPT_CHEF), HumanMessage("\n".join(contexto))]
        )
    except (ValidationError, OutputParserException) as erro:
        # Devolve o erro ao agente para nova tentativa
        return (
            f"ERRO: a receita não passou na validação do schema ({erro}). Tente novamente.",
            {"receita": None, "erro_validacao": str(erro)},
        )

    resumo = (
        f"Receita criada e validada: '{receita.nome}' ({receita.tempo_preparo_min} min, "
        f"{receita.porcoes} porções). Prioritários usados: "
        f"{', '.join(receita.ingredientes_prioritarios_usados) or 'nenhum'}. "
        "O app já exibe a receita completa ao usuário."
    )
    return resumo, {"receita": receita.model_dump(), "erro_validacao": None}


TOOLS = [check_geladeira, verificar_validade, buscar_receitas_similares, criar_receita]
NOMES_TOOLS = [t.name for t in TOOLS]
