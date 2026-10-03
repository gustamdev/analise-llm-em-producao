"""Geladeira virtual: sorteio de ingredientes, quantidades e validades."""

import random
from datetime import date, timedelta

from .schemas import ItemGeladeira

# categoria -> (faixa de validade em dias, [(ingrediente, unidade), ...])
CATALOGO: dict[str, tuple[tuple[int, int], list[tuple[str, str]]]] = {
    "laticínios": ((4, 25), [
        ("leite integral", "ml"), ("iogurte natural", "g"), ("queijo muçarela", "g"),
        ("queijo minas", "g"), ("queijo parmesão", "g"), ("manteiga", "g"),
        ("creme de leite", "g"), ("requeijão", "g"), ("ricota", "g"), ("cream cheese", "g"),
        ("nata", "g"), ("queijo coalho", "g"),
    ]),
    "carnes": ((1, 6), [
        ("peito de frango", "g"), ("coxa de frango", "g"), ("carne moída", "g"),
        ("bife de alcatra", "g"), ("linguiça toscana", "g"), ("bacon", "g"),
        ("lombo suíno", "g"), ("presunto", "g"), ("peito de peru", "g"), ("costela bovina", "g"),
    ]),
    "peixes e frutos do mar": ((1, 4), [
        ("filé de tilápia", "g"), ("salmão", "g"), ("camarão", "g"),
        ("bacalhau dessalgado", "g"), ("sardinha fresca", "g"),
    ]),
    "ovos e proteínas vegetais": ((5, 30), [
        ("ovos", "un"), ("tofu", "g"), ("grão-de-bico cozido", "g"), ("lentilha cozida", "g"),
    ]),
    "hortaliças": ((3, 14), [
        ("alface", "un"), ("tomate", "un"), ("cebola", "un"), ("alho", "cabeça"),
        ("cenoura", "un"), ("abobrinha", "un"), ("berinjela", "un"),
        ("pimentão vermelho", "un"), ("pimentão verde", "un"), ("brócolis", "maço"),
        ("couve-flor", "un"), ("espinafre", "maço"), ("couve", "maço"), ("batata", "un"),
        ("batata-doce", "un"), ("mandioca", "g"), ("abóbora cabotiá", "g"), ("pepino", "un"),
        ("repolho", "un"), ("rúcula", "maço"), ("cogumelo paris", "g"), ("milho verde", "un"),
        ("vagem", "g"), ("quiabo", "g"), ("chuchu", "un"), ("beterraba", "un"),
        ("alho-poró", "un"), ("ervilha fresca", "g"), ("rabanete", "un"),
    ]),
    "frutas": ((2, 12), [
        ("banana", "un"), ("maçã", "un"), ("limão", "un"), ("laranja", "un"),
        ("morango", "g"), ("mamão", "un"), ("manga", "un"), ("abacate", "un"),
        ("abacaxi", "un"), ("uva", "g"), ("maracujá", "un"), ("pera", "un"),
    ]),
    "ervas frescas": ((2, 7), [
        ("salsinha", "maço"), ("cebolinha", "maço"), ("coentro", "maço"),
        ("manjericão", "maço"), ("hortelã", "maço"), ("alecrim", "maço"), ("gengibre", "g"),
    ]),
    "molhos e condimentos": ((15, 90), [
        ("mostarda", "g"), ("ketchup", "g"), ("maionese", "g"), ("molho de tomate", "g"),
        ("molho shoyu", "ml"), ("pesto", "g"), ("leite de coco", "ml"), ("extrato de tomate", "g"),
    ]),
    "prontos e sobras": ((1, 5), [
        ("arroz cozido", "g"), ("feijão cozido", "g"), ("massa fresca", "g"),
        ("purê de batata", "g"), ("frango desfiado", "g"), ("caldo de legumes caseiro", "ml"),
        ("massa de pizza", "g"), ("pão de forma", "un"), ("tortilla de trigo", "un"),
    ]),
    "bebidas e outros": ((5, 60), [
        ("suco de laranja", "ml"), ("vinho branco aberto", "ml"), ("cerveja", "ml"),
        ("água de coco", "ml"), ("chocolate meio amargo", "g"), ("geleia de morango", "g"),
    ]),
}

# unidade -> (mínimo, máximo, passo) da quantidade
FAIXAS_QUANTIDADE = {
    "g": (100, 1000, 50),
    "ml": (200, 1000, 100),
    "un": (1, 12, 1),
    "maço": (1, 2, 1),
    "cabeça": (1, 3, 1),
}

TOTAL_INGREDIENTES_CATALOGO = sum(len(itens) for _, itens in CATALOGO.values())


def gerar_geladeira(
    n_itens: int = 20,
    seed: int | None = None,
    data_referencia: date | None = None,
    fracao_perto_de_vencer: float = 0.3,
    so_itens_frescos: bool = False,
) -> dict:
    """Sorteia uma geladeira com `n_itens` ingredientes, com seed opcional."""
    if seed is None:
        seed = random.randrange(1_000_000)  # sem seed: geladeira nova
    rng = random.Random(seed)  # gerador isolado do random global
    data_ref = data_referencia or date.today()

    todos = [(nome, unidade, cat) for cat, (_, itens) in CATALOGO.items() for nome, unidade in itens]
    escolhidos = rng.sample(todos, min(n_itens, len(todos)))

    itens = []
    for nome, unidade, categoria in escolhidos:
        dias_min, dias_max = CATALOGO[categoria][0]
        if so_itens_frescos:
            dias = rng.randint(max(dias_min, 4), max(dias_max, 4))
        elif rng.random() < fracao_perto_de_vencer:
            dias = rng.randint(-2, 2)  # de 2 dias vencido a 2 dias para vencer
        else:
            dias = rng.randint(dias_min, dias_max)
        minimo, maximo, passo = FAIXAS_QUANTIDADE[unidade]
        quantidade = rng.randrange(minimo, maximo + 1, passo)
        itens.append(
            ItemGeladeira(
                nome=nome,
                quantidade=quantidade,
                unidade=unidade,
                categoria=categoria,
                validade=(data_ref + timedelta(days=dias)).isoformat(),
            ).model_dump()
        )

    itens.sort(key=lambda i: (i["categoria"], i["nome"]))
    return {"seed": seed, "data_referencia": data_ref.isoformat(), "itens": itens}


def dias_para_vencer(item: dict, data_referencia: str) -> int:
    return (date.fromisoformat(item["validade"]) - date.fromisoformat(data_referencia)).days


def classificar_validade(geladeira: dict, dias_limite: int = 3) -> dict:
    """Separa os itens em `vencidos` e `vencendo` (até `dias_limite` dias)."""
    ref = geladeira["data_referencia"]
    vencidos, vencendo = [], []
    for item in geladeira["itens"]:
        dias = dias_para_vencer(item, ref)
        registro = {**item, "dias_para_vencer": dias}
        if dias < 0:
            vencidos.append(registro)
        elif dias <= dias_limite:
            vencendo.append(registro)
    vencendo.sort(key=lambda i: i["dias_para_vencer"])
    return {"vencidos": vencidos, "vencendo": vencendo}


def formatar_item(item: dict) -> str:
    """Formata o item em uma linha curta para o prompt."""
    qtd = int(item["quantidade"]) if float(item["quantidade"]).is_integer() else item["quantidade"]
    return f"- {item['nome']}: {qtd} {item['unidade']} (validade {item['validade']})"
