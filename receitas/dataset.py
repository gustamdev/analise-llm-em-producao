"""Golden dataset: geração, congelamento e publicação no LangSmith."""

import json
from datetime import date

from langsmith import Client

from .config import ARQ_GOLDEN, NOME_DATASET
from .geladeira import classificar_validade, gerar_geladeira

CHECK, VALIDADE, BUSCA, CRIAR = "check_geladeira", "verificar_validade", "buscar_receitas_similares", "criar_receita"

# Cenários: pergunta, tools esperadas, tamanho da geladeira, fração vencendo e histórico da v0
CENARIOS = [
    # Fácil: geladeira pequena, pedido direto
    dict(id="F1", complexidade="facil", tools=[CHECK, CRIAR], n_itens=(6, 10), fracao=0.0,
         pergunta="O que posso cozinhar hoje com o que tenho na geladeira?",
         nota="Uma receita simples e viável só com itens da geladeira.",
         historico=("acerto", "")),
    dict(id="F2", complexidade="facil", tools=[CHECK, CRIAR], n_itens=(6, 10), fracao=0.0,
         pergunta="Me sugere um jantar rápido, de uns 20 minutos, com o que tem aqui.",
         nota="Receita de até ~25 minutos com itens da geladeira.",
         historico=("acerto", "")),
    dict(id="F3", complexidade="facil", tools=[CHECK, CRIAR], n_itens=(6, 10), fracao=0.0,
         pergunta="Quero um café da manhã reforçado com o que tenho na geladeira.",
         nota="Receita adequada para café da manhã com itens da geladeira.",
         historico=("erro", "v0 chamou buscar_receitas_similares sem necessidade (custo ~3x maior)")),
    dict(id="F4", complexidade="facil", tools=[VALIDADE], n_itens=(8, 12), fracao=0.4,
         pergunta="Quais itens da minha geladeira estão vencendo?",
         nota="Listar itens vencidos (descartar) e os que vencem logo. Não precisa de receita.",
         historico=("erro", "v0 chamou check_geladeira + verificar_validade (redundante)")),
    dict(id="F5", complexidade="facil", tools=[], n_itens=(6, 8), fracao=0.0,
         pergunta="Qual a temperatura ideal para manter a geladeira?",
         nota="Entre 1 °C e 4 °C (nunca acima de 5 °C). Pergunta geral: nenhuma tool é necessária.",
         historico=("acerto", "")),
    dict(id="F6", complexidade="facil", tools=[CHECK, CRIAR], n_itens=(6, 10), fracao=0.0,
         pergunta="Dá pra fazer um lanche pra levar pro trabalho com o que tem na geladeira?",
         nota="Lanche transportável feito com itens da geladeira.",
         historico=("acerto", "")),
    # Médio: priorizar o que vence
    dict(id="M1", complexidade="medio", tools=[VALIDADE, CRIAR], n_itens=(15, 25), fracao=0.3,
         pergunta="Faz uma receita só com o que está para vencer, não quero desperdiçar nada.",
         nota="Receita centrada nos itens que vencem logo, sem itens vencidos.",
         historico=("acerto", "")),
    dict(id="M2", complexidade="medio", tools=[VALIDADE, CRIAR], n_itens=(15, 25), fracao=0.35,
         pergunta="Me ajuda a usar o que vence nos próximos dias num almoço?",
         nota="Almoço que aproveita os itens que vencem logo; não pode usar vencidos.",
         historico=("erro", "v0 usou um item JÁ vencido na receita")),
    dict(id="M3", complexidade="medio", tools=[CHECK, VALIDADE, CRIAR], n_itens=(15, 25), fracao=0.3,
         pergunta="Quero um jantar que aproveite o que está vencendo, mas pode completar com outras coisas da geladeira.",
         nota="Jantar que usa os itens que vencem logo, complementado por outros itens da geladeira.",
         historico=("acerto", "")),
    dict(id="M4", complexidade="medio", tools=[VALIDADE, CRIAR], n_itens=(15, 25), fracao=0.35,
         pergunta="Tenho coisas quase estragando aqui, monta uma sopa com elas?",
         nota="Sopa feita com os itens que vencem logo (se fizer sentido culinário).",
         historico=("erro", "v0 ignorou a validade e montou a sopa com itens que duram semanas")),
    dict(id="M5", complexidade="medio", tools=[VALIDADE, CRIAR], n_itens=(15, 25), fracao=0.3,
         pergunta="Antes de ir ao mercado quero gastar o que vence logo. Uma receita para 4 pessoas, por favor.",
         nota="Receita para 4 porções priorizando o que vence logo.",
         historico=("acerto", "")),
    # Difícil: RAG + inventário + validade, com restrições
    dict(id="D1", complexidade="dificil", tools=[BUSCA, CHECK, VALIDADE, CRIAR], n_itens=(35, 60), fracao=0.25,
         pergunta="Quero algo no estilo de um risoto, inspirado em receitas clássicas, usando o que vence logo e o resto da geladeira.",
         nota="Prato estilo risoto inspirado em receitas reais, adaptado à geladeira, priorizando o que vence.",
         historico=("acerto", "")),
    dict(id="D2", complexidade="dificil", tools=[BUSCA, CHECK, VALIDADE, CRIAR], n_itens=(35, 60), fracao=0.25,
         pergunta="Vou receber amigos para jantar: me inspira numa receita conhecida de curry, priorizando o que está vencendo.",
         nota="Curry inspirado em receitas reais; se faltar algo típico (ex.: leite de coco), adaptar e avisar.",
         historico=("erro", "v0 sugeriu curry com leite de coco que não estava na geladeira")),
    dict(id="D3", complexidade="dificil", tools=[BUSCA, CHECK, VALIDADE, CRIAR], n_itens=(35, 60), fracao=0.25,
         pergunta="Quero uma lasanha vegetariana parecida com receitas tradicionais, aproveitando o que vai estragar. Se faltar algo, me diga.",
         nota="Lasanha SEM carne, inspirada em receitas reais; deve dizer claramente o que falta.",
         historico=("erro", "v0 inventou massa de lasanha sem avisar que ela não estava na geladeira")),
    dict(id="D4", complexidade="dificil", tools=[BUSCA, CHECK, VALIDADE, CRIAR], n_itens=(35, 60), fracao=0.25,
         pergunta="Me mostra receitas parecidas com shakshuka e adapta pro que eu tenho, usando primeiro o que está vencendo.",
         nota="Adaptação de shakshuka (ovos + molho de tomate/pimentão) aos itens disponíveis.",
         historico=("acerto", "")),
    dict(id="D5", complexidade="dificil", tools=[BUSCA, CHECK, VALIDADE, CRIAR], n_itens=(35, 60), fracao=0.25,
         pergunta="Quero um prato sem carne, inspirado em receitas de stir-fry, usando o que vence e o que mais tiver na geladeira.",
         nota="Stir-fry SEM carne/peixe, inspirado em receitas reais, priorizando o que vence.",
         historico=("acerto", "")),
]


def _geladeira_para_cenario(cenario: dict, n_itens: int, seed: int, data_ref: date) -> tuple[dict, dict]:
    """Sorteia geladeiras até achar uma coerente com o cenário."""
    fresca = cenario["fracao"] == 0
    for tentativa in range(50):
        geladeira = gerar_geladeira(n_itens, seed=seed + 1000 * tentativa, data_referencia=data_ref,
                                    fracao_perto_de_vencer=cenario["fracao"], so_itens_frescos=fresca)
        validade = classificar_validade(geladeira, dias_limite=3)
        if fresca or (validade["vencidos"] and len(validade["vencendo"]) >= 2):
            return geladeira, validade
    raise RuntimeError(f"Não achei geladeira coerente para o cenário {cenario['id']}")


def gerar_golden_dataset(seed: int = 2026, data_referencia: date | None = None) -> list[dict]:
    """Gera os exemplos com seed fixa."""
    data_ref = data_referencia or date.today()
    exemplos = []
    for indice, cenario in enumerate(CENARIOS):
        n_min, n_max = cenario["n_itens"]
        n_itens = n_min + (seed + indice) % (n_max - n_min + 1)
        geladeira, validade = _geladeira_para_cenario(cenario, n_itens, seed + indice, data_ref)

        # Gabarito calculado pelas datas, sem LLM
        prioritarios = [i["nome"] for i in validade["vencendo"]]
        vencidos = [i["nome"] for i in validade["vencidos"]]
        status, observacao = cenario["historico"]

        exemplos.append({
            "inputs": {"pergunta": cenario["pergunta"], "geladeira": geladeira},
            "outputs": {
                "id_cenario": cenario["id"],
                "tools_esperadas": cenario["tools"],
                "complexidade": cenario["complexidade"],
                "ingredientes_prioritarios": prioritarios,
                "ingredientes_vencidos": vencidos,
                "resposta_referencia": (
                    f"{cenario['nota']} Prioritários: {prioritarios or 'nenhum'}. "
                    f"Proibidos (vencidos): {vencidos or 'nenhum'}."
                ),
                "historico_status": status,
                "historico_observacao": observacao,
            },
            "metadata": {
                "id_cenario": cenario["id"],
                "complexidade": cenario["complexidade"],
                "historico_status": status,
                "n_itens_geladeira": n_itens,
            },
            "split": cenario["complexidade"],
        })
    return exemplos


def carregar_ou_criar_golden(forcar_recriacao: bool = False) -> list[dict]:
    """Carrega o JSON congelado ou gera na primeira execução."""
    if ARQ_GOLDEN.exists() and not forcar_recriacao:
        return json.loads(ARQ_GOLDEN.read_text(encoding="utf-8"))
    exemplos = gerar_golden_dataset()
    ARQ_GOLDEN.write_text(json.dumps(exemplos, ensure_ascii=False, indent=1), encoding="utf-8")
    return exemplos


def publicar_no_langsmith(exemplos: list[dict], nome: str = NOME_DATASET) -> str:
    """Publica o dataset no LangSmith se ainda não existir."""
    client = Client()
    if client.has_dataset(dataset_name=nome):
        return f"Dataset '{nome}' já existe no LangSmith — mantido congelado."
    client.create_dataset(
        dataset_name=nome,
        description="Golden dataset do assistente de receitas: casos validados (acertos e erros "
                    "históricos) com tools esperadas e nível de complexidade.",
    )
    client.create_examples(dataset_name=nome, examples=exemplos)
    return f"Dataset '{nome}' criado com {len(exemplos)} exemplos."
