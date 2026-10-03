"""Configuração do projeto: modelos, preços, caminhos e nomes no LangSmith."""

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"
DATA_DIR.mkdir(exist_ok=True)

# Carrega as chaves do .env
load_dotenv(RAIZ / ".env")

# LangSmith
PROJETO_LANGSMITH = os.getenv("LANGSMITH_PROJECT", "assistente-receitas")
os.environ.setdefault("LANGSMITH_PROJECT", PROJETO_LANGSMITH)
NOME_DATASET = "receitas-golden-v1"  # dataset congelado; novo critério = nova versão

# Modelos
# gpt-4.1: aceita temperature=0 e tem latência estável
MODELO_FRONTEIRA = "gpt-4.1"
MODELO_INTERMEDIARIO = "gpt-4.1-mini"
MODELO_SIMPLES = "gpt-4.1-nano"
MODELOS_CANDIDATOS = [MODELO_FRONTEIRA, MODELO_INTERMEDIARIO, MODELO_SIMPLES]

MODELO_ROTEADOR = MODELO_SIMPLES  # classificação simples, modelo barato
MODELO_JUIZ = MODELO_FRONTEIRA  # mesmo juiz em todos os experimentos
MODELO_EMBEDDING = "text-embedding-3-small"

# US$ por 1M tokens (entrada, saída); conferir em openai.com/api/pricing
PRECOS_POR_MILHAO = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "text-embedding-3-small": (0.02, 0.0),
}


def custo_usd(modelo: str, tokens_entrada: int, tokens_saida: int = 0) -> float:
    """Converte tokens em US$."""
    # Aceita nomes com data, ex.: "gpt-4.1-mini-2025-04-14"
    chave = max((m for m in PRECOS_POR_MILHAO if modelo.startswith(m)), key=len, default=None)
    if chave is None:
        return 0.0
    preco_in, preco_out = PRECOS_POR_MILHAO[chave]
    return (tokens_entrada * preco_in + tokens_saida * preco_out) / 1_000_000


# RAG
# ~13,5 mil receitas do Epicurious (CC-BY-SA 3.0), via Hugging Face
URL_DATASET_RECEITAS = (
    "https://huggingface.co/api/datasets/Hieu-Pham/kaggle_food_recipes/"
    "parquet/default/train/0.parquet"
)
# Amostra indexada no Chroma
N_RECEITAS_INDEXADAS = 2000
CHROMA_DIR = DATA_DIR / "chroma"

# Arquivos gerados
ARQ_GOLDEN = DATA_DIR / "golden_dataset.json"
ARQ_HISTORICO = DATA_DIR / "historico_avaliacoes.csv"
ARQ_RUNS = DATA_DIR / "runs_export.csv"
ARQ_POLITICA = DATA_DIR / "politica_roteamento.json"
