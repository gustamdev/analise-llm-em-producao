"""Funções de nota usadas no `evaluate()`."""

import json
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from .config import MODELO_JUIZ
from .geladeira import formatar_item
from .llm import obter_llm
from .schemas import AvaliacaoJuiz, Receita


def _espera_receita(reference_outputs: dict) -> bool:
    return "criar_receita" in reference_outputs["tools_esperadas"]


# Estrutural (Pydantic)
def schema_valido(outputs: dict, reference_outputs: dict) -> dict:
    """Confere se a receita bate com o schema `Receita`."""
    if not _espera_receita(reference_outputs):
        return {"key": "schema_valido", "score": None, "comment": "pergunta não pede receita"}
    if outputs.get("receita") is None:
        return {"key": "schema_valido", "score": 0, "comment": outputs.get("erro_validacao") or "sem receita"}
    try:
        Receita.model_validate(outputs["receita"])
        return {"key": "schema_valido", "score": 1}
    except ValidationError as erro:
        return {"key": "schema_valido", "score": 0, "comment": str(erro)[:300]}


# Tool selection
def tools_exatas(outputs: dict, reference_outputs: dict) -> dict:
    """Checa se as tools chamadas batem exatamente com as esperadas."""
    esperadas, chamadas = set(reference_outputs["tools_esperadas"]), set(outputs["tools_chamadas"])
    faltaram, sobraram = esperadas - chamadas, chamadas - esperadas
    comentario = f"faltaram={sorted(faltaram)} sobraram={sorted(sobraram)}" if (faltaram or sobraram) else "ok"
    return {"key": "tools_exatas", "score": int(esperadas == chamadas), "comment": comentario}


def tools_f1(outputs: dict, reference_outputs: dict) -> dict:
    """F1 entre tools esperadas e chamadas (crédito parcial)."""
    esperadas, chamadas = set(reference_outputs["tools_esperadas"]), set(outputs["tools_chamadas"])
    if not esperadas and not chamadas:
        return {"key": "tools_f1", "score": 1.0}
    acertos = len(esperadas & chamadas)
    precisao = acertos / len(chamadas) if chamadas else 0.0
    recall = acertos / len(esperadas) if esperadas else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if acertos else 0.0
    return {"key": "tools_f1", "score": round(f1, 3)}


# Regra de negócio
def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


def aproveitamento_prioritarios(outputs: dict, reference_outputs: dict) -> dict:
    """Conta quantos itens perto do vencimento aparecem na receita."""
    prioritarios = reference_outputs.get("ingredientes_prioritarios", [])
    if not _espera_receita(reference_outputs) or not prioritarios:
        return {"key": "aproveitamento_prioritarios", "score": None}
    if not outputs.get("receita"):
        return {"key": "aproveitamento_prioritarios", "score": 0.0}
    texto = _normalizar(" ".join(outputs["receita"]["ingredientes"]))
    # Conta o item se alguma palavra do nome aparece nos ingredientes
    usados = [p for p in prioritarios if any(w in texto for w in _normalizar(p).split() if len(w) >= 3)]
    # Nota máxima com 4 itens usados (ou todos, se houver menos)
    meta = min(len(prioritarios), 4)
    return {
        "key": "aproveitamento_prioritarios",
        "score": round(min(len(usados), meta) / meta, 3),
        "comment": f"usou {usados} de {prioritarios}",
    }


# LLM-as-judge
PROMPT_JUIZ = """Você é um avaliador rigoroso de um assistente de receitas.
Avalie a resposta com notas de 1 a 5 em cada critério, seguindo a rubrica:
- uso_da_geladeira: 5 = só usa itens que existem na geladeira (+ sal, pimenta, óleo, azeite,
  açúcar, farinha, água); 1 = depende de vários ingredientes que o usuário não tem.
- prioriza_vencimento: 5 = aproveita os itens que vencem logo e NÃO usa itens vencidos;
  1 = usa item vencido OU ignora todos os prioritários.
- coerencia_preparo: 5 = passos claros, ordem lógica, tempo e porções plausíveis.
- atende_pedido: 5 = faz exatamente o que o usuário pediu (estilo, restrições, ocasião).
Seja objetivo: compare com os FATOS fornecidos, não com o que você acha que deveria haver."""


def _montar_caso_para_juiz(inputs: dict, outputs: dict, reference_outputs: dict) -> str:
    geladeira = inputs["geladeira"]
    return "\n".join([
        f"PEDIDO DO USUÁRIO: {inputs['pergunta']}",
        f"\nGELADEIRA (hoje = {geladeira['data_referencia']}):",
        *[formatar_item(i) for i in geladeira["itens"]],
        f"\nFATOS: itens que vencem em até 3 dias = {reference_outputs.get('ingredientes_prioritarios', [])}",
        f"FATOS: itens VENCIDOS (não podem ser usados) = {reference_outputs.get('ingredientes_vencidos', [])}",
        f"CRITÉRIO DE REFERÊNCIA: {reference_outputs.get('resposta_referencia', '')}",
        f"\nRECEITA GERADA (JSON):\n{json.dumps(outputs.get('receita'), ensure_ascii=False, indent=1)}",
        f"\nRESPOSTA FINAL AO USUÁRIO:\n{outputs.get('resposta', '')}",
    ])


def julgar(inputs: dict, outputs: dict, reference_outputs: dict) -> AvaliacaoJuiz:
    """Envia o caso ao juiz e retorna a avaliação estruturada."""
    juiz = obter_llm(MODELO_JUIZ).with_structured_output(AvaliacaoJuiz, method="function_calling")
    return juiz.invoke([
        SystemMessage(PROMPT_JUIZ),
        HumanMessage(_montar_caso_para_juiz(inputs, outputs, reference_outputs)),
    ])


def juiz_qualidade(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Nota 0..1 do juiz para a receita."""
    if not _espera_receita(reference_outputs):
        return {"key": "juiz_qualidade", "score": None, "comment": "pergunta não pede receita"}
    if not outputs.get("receita"):
        return {"key": "juiz_qualidade", "score": 0.0, "comment": "nenhuma receita foi entregue"}
    avaliacao = julgar(inputs, outputs, reference_outputs)
    return {"key": "juiz_qualidade", "score": avaliacao.nota(), "comment": avaliacao.raciocinio}


# Roteamento
def roteamento_correto(outputs: dict, reference_outputs: dict) -> dict:
    """Confere se o roteador acertou o nível do pedido."""
    if outputs.get("complexidade_roteada") is None:
        return {"key": "roteamento_correto", "score": None}
    return {
        "key": "roteamento_correto",
        "score": int(outputs["complexidade_roteada"] == reference_outputs["complexidade"]),
        "comment": f"roteou={outputs['complexidade_roteada']} esperado={reference_outputs['complexidade']}",
    }


AVALIADORES_DETERMINISTICOS = [schema_valido, tools_exatas, tools_f1, aproveitamento_prioritarios]
AVALIADORES_PADRAO = [*AVALIADORES_DETERMINISTICOS, juiz_qualidade]
