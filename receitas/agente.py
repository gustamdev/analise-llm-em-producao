"""Grafo do agente: roteador opcional + loop de tools."""

import time
from typing import Annotated, Callable, TypedDict

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .config import MODELO_FRONTEIRA, MODELO_ROTEADOR, custo_usd
from .llm import obter_llm
from .schemas import ClassificacaoComplexidade
from .tools import TOOLS

PROMPTS_SISTEMA = {
    # v1: prompt completo (baseline)
    "v1": """Você é um assistente de culinária que ajuda o usuário a cozinhar com o que tem na geladeira.

## Como escolher as tools (use o MÍNIMO necessário: cada chamada custa tempo e dinheiro)
- check_geladeira: quando precisar do inventário completo.
- verificar_validade: quando o pedido fala em aproveitar, não desperdiçar ou usar o que vence.
  Se o pedido é SÓ sobre os itens que vencem, ela basta (não chame check_geladeira).
- buscar_receitas_similares: SOMENTE se o usuário pedir inspiração num prato/estilo específico
  ou receitas conhecidas. Escreva a consulta em inglês.
- criar_receita: sempre que for entregar uma receita. Chame UMA vez, passando ingredientes
  reais da geladeira (nunca vencidos) e os prioritários, se houver.
- Perguntas gerais de cozinha que não dependem da geladeira: responda direto, sem tools.

## Regras
- Nunca sugira itens vencidos.
- Não invente ingredientes fora da geladeira (exceto sal, pimenta, óleo, azeite, açúcar, farinha, água).
- Se faltar algo essencial, diga com honestidade e proponha a melhor adaptação possível.

## Resposta final
Curta (até 4 frases), em português: nome do prato, por que atende ao pedido e quais itens
perto do vencimento foram aproveitados. Não repita a receita inteira: o app já a exibe.""",
    # v2_enxuto: prompt reduzido para economizar tokens
    "v2_enxuto": """Você é um assistente de culinária. Use as tools para ajudar o usuário
a cozinhar com o que tem na geladeira. Responda em português, de forma curta.""",
}

PROMPT_ROTEADOR = """Classifique a complexidade de um pedido para um assistente de receitas.
- facil: pedido direto ("o que faço com o que tenho?"), dúvida geral de cozinha, ou só listar o que vence.
- medio: pede uma receita priorizando/aproveitando o que está para vencer.
- dificil: pede inspiração num prato/estilo específico ou receitas conhecidas, OU combina
  restrições (dieta, convidados, ingredientes que faltam) com aproveitar o que vence."""


class EstadoAgente(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # histórico de mensagens
    geladeira: dict  # injetada nas tools
    modelo: str  # fixo ou escolhido pelo roteador
    complexidade: str | None  # preenchida pelo roteador


def construir_agente(versao_prompt: str = "v1", politica: dict[str, str] | None = None):
    """Monta e compila o grafo, com roteador quando há política."""
    prompt_sistema = PROMPTS_SISTEMA[versao_prompt]

    def roteador(estado: EstadoAgente) -> dict:
        pergunta = estado["messages"][-1].content
        classificador = obter_llm(MODELO_ROTEADOR).with_structured_output(
            ClassificacaoComplexidade, method="function_calling"
        )
        resultado = classificador.invoke([
            SystemMessage(PROMPT_ROTEADOR),
            HumanMessage(f"Pedido: {pergunta}\nItens na geladeira: {len(estado['geladeira']['itens'])}"),
        ])
        return {"complexidade": resultado.nivel, "modelo": politica[resultado.nivel]}

    def agente(estado: EstadoAgente) -> dict:
        llm = obter_llm(estado["modelo"]).bind_tools(TOOLS)
        resposta = llm.invoke([SystemMessage(prompt_sistema), *estado["messages"]])
        return {"messages": [resposta]}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("agente", agente)
    grafo.add_node("tools", ToolNode(TOOLS))  # erros das tools voltam como mensagem
    if politica:
        grafo.add_node("roteador", roteador)
        grafo.add_edge(START, "roteador")
        grafo.add_edge("roteador", "agente")
    else:
        grafo.add_edge(START, "agente")
    grafo.add_conditional_edges("agente", tools_condition)  # tem tool_call? -> "tools" : END
    grafo.add_edge("tools", "agente")
    return grafo.compile()


def resumir_execucao(estado: dict) -> dict:
    """Resume o estado final do agente num dict simples."""
    mensagens = estado["messages"]
    tools_chamadas = [c["name"] for m in mensagens if isinstance(m, AIMessage) for c in m.tool_calls]
    resultados_tools = [m for m in mensagens if isinstance(m, ToolMessage)]

    receita, erro_validacao = None, None
    for m in resultados_tools:
        if m.name == "criar_receita" and m.artifact:
            receita, erro_validacao = m.artifact["receita"], m.artifact["erro_validacao"]
    custo_embedding = sum(
        (m.artifact or {}).get("custo_embedding_usd", 0.0)
        for m in resultados_tools
        if m.name == "buscar_receitas_similares"
    )
    return {
        "resposta": mensagens[-1].content,
        "receita": receita,
        "erro_validacao": erro_validacao,
        "tools_chamadas": tools_chamadas,
        "modelo": estado["modelo"],
        "complexidade_roteada": estado.get("complexidade"),
        "custo_embedding_usd": custo_embedding,
    }


def criar_alvo(
    modelo: str = MODELO_FRONTEIRA,
    versao_prompt: str = "v1",
    politica: dict[str, str] | None = None,
    max_passos: int = 12,
) -> Callable[[dict], dict]:
    """Monta a função que o `evaluate()` chama a cada exemplo."""
    agente = construir_agente(versao_prompt, politica)
    rotulo_modelo = "roteado" if politica else modelo

    def alvo(inputs: dict) -> dict:
        inicio = time.perf_counter()
        config = {
            "recursion_limit": max_passos,  # limite de passos do agente
            "run_name": "assistente_receitas",
            "tags": [f"prompt:{versao_prompt}", f"modelo:{rotulo_modelo}"],
            "metadata": {"versao_prompt": versao_prompt, "modelo_config": rotulo_modelo},
        }
        entrada = {
            "messages": [HumanMessage(inputs["pergunta"])],
            "geladeira": inputs["geladeira"],
            "modelo": modelo,
            "complexidade": None,
        }
        # Soma os tokens de todas as chamadas de LLM, por modelo
        with get_usage_metadata_callback() as uso:
            try:
                saida = resumir_execucao(agente.invoke(entrada, config))
            except GraphRecursionError:
                saida = {"resposta": "ERRO: limite de passos atingido", "receita": None,
                         "erro_validacao": None, "tools_chamadas": [], "modelo": modelo,
                         "complexidade_roteada": None, "custo_embedding_usd": 0.0}

        tokens_in = sum(u["input_tokens"] for u in uso.usage_metadata.values())
        tokens_out = sum(u["output_tokens"] for u in uso.usage_metadata.values())
        custo_llm = sum(
            custo_usd(nome, u["input_tokens"], u["output_tokens"]) for nome, u in uso.usage_metadata.items()
        )
        return {
            **saida,
            "tokens_entrada": tokens_in,
            "tokens_saida": tokens_out,
            "custo_usd": round(custo_llm + saida["custo_embedding_usd"], 6),
            "latencia_s": round(time.perf_counter() - inicio, 2),
        }

    return alvo
