"""RAG sobre as receitas reais, indexadas no Chroma."""

import ast
from functools import lru_cache

import pandas as pd
import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langsmith import get_current_run_tree, traceable

from .config import (
    CHROMA_DIR,
    DATA_DIR,
    MODELO_EMBEDDING,
    N_RECEITAS_INDEXADAS,
    URL_DATASET_RECEITAS,
    custo_usd,
)

ARQ_AMOSTRA = DATA_DIR / "receitas_amostra.parquet"
_TOKENIZER = tiktoken.get_encoding("cl100k_base")  # tokenizer dos modelos text-embedding-3


def contar_tokens(texto: str) -> int:
    return len(_TOKENIZER.encode(texto))


def carregar_receitas(n: int = N_RECEITAS_INDEXADAS, seed: int = 42) -> pd.DataFrame:
    """Baixa o dataset público (12 MB) e guarda uma amostra fixa de `n` receitas."""
    if ARQ_AMOSTRA.exists():
        return pd.read_parquet(ARQ_AMOSTRA)
    df = pd.read_parquet(URL_DATASET_RECEITAS, columns=["Title", "Cleaned_Ingredients", "Instructions"])
    df = df.dropna()
    df = df[(df["Title"].str.strip() != "") & (df["Instructions"].str.len() > 50)]
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    # Converte o texto da lista Python em string separada por ';'
    df["ingredientes"] = df["Cleaned_Ingredients"].apply(lambda s: "; ".join(ast.literal_eval(s)))
    df = df.rename(columns={"Title": "titulo", "Instructions": "instrucoes"})
    df = df[["titulo", "ingredientes", "instrucoes"]]
    df.to_parquet(ARQ_AMOSTRA)
    return df


def _para_documento(linha: pd.Series) -> Document:
    # Embedding de título + ingredientes; instruções nos metadados
    return Document(
        page_content=f"{linha['titulo']}\nIngredients: {linha['ingredientes']}",
        metadata={"titulo": linha["titulo"], "instrucoes": linha["instrucoes"][:800]},
    )


@lru_cache(maxsize=1)
def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=MODELO_EMBEDDING)


@lru_cache(maxsize=1)
def obter_indice() -> Chroma:
    return Chroma(
        collection_name="receitas",
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def estimar_custo_indexacao(df: pd.DataFrame) -> dict:
    tokens = sum(contar_tokens(_para_documento(l).page_content) for _, l in df.iterrows())
    return {"documentos": len(df), "tokens": tokens, "custo_usd": custo_usd(MODELO_EMBEDDING, tokens)}


@traceable(
    run_type="embedding",
    name="indexar_receitas",
    metadata={"ls_provider": "openai", "ls_model_name": MODELO_EMBEDDING},
    process_inputs=lambda entradas: {"documentos": len(entradas["df"])},
)
def construir_indice(df: pd.DataFrame, lote: int = 500) -> dict:
    """Cria o índice (só se estiver vazio) e registra o custo da indexação no LangSmith."""
    indice = obter_indice()
    if indice.get(limit=1)["ids"]:
        return {"status": "índice já existia, nada a pagar", "documentos": len(indice.get()["ids"])}

    docs = [_para_documento(l) for _, l in df.iterrows()]
    for inicio in range(0, len(docs), lote):
        parte = docs[inicio : inicio + lote]
        indice.add_documents(parte, ids=[str(i) for i in range(inicio, inicio + len(parte))])

    estimativa = estimar_custo_indexacao(df)
    _registrar_uso_embedding(estimativa["tokens"])
    return {"status": "índice criado", **estimativa}


def _registrar_uso_embedding(tokens: int) -> float:
    """Registra tokens e custo do embedding no run atual do LangSmith."""
    custo = custo_usd(MODELO_EMBEDDING, tokens)
    run = get_current_run_tree()
    if run is not None:
        run.set(usage_metadata={
            "input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens,
            "input_cost": custo, "output_cost": 0.0, "total_cost": custo,
        })
    return custo


@traceable(
    run_type="embedding",
    name="embedding_da_consulta",
    metadata={"ls_provider": "openai", "ls_model_name": MODELO_EMBEDDING},
    process_outputs=lambda saida: {"dimensoes": len(saida[0]), "tokens": saida[1]},
)
def _embedding_consulta(texto: str) -> tuple[list[float], int]:
    vetor = _embeddings().embed_query(texto)
    tokens = contar_tokens(texto)
    _registrar_uso_embedding(tokens)
    return vetor, tokens


def _saida_retriever(saida: tuple[list[Document], dict]) -> dict:
    """Formata a saída no padrão de retriever do LangSmith."""
    docs, uso = saida
    return {
        "documents": [{"type": "Document", "page_content": d.page_content, "metadata": d.metadata} for d in docs],
        **uso,
    }


@traceable(run_type="retriever", name="busca_vetorial_chroma", process_outputs=_saida_retriever)
def buscar_receitas(consulta: str, k: int = 3) -> tuple[list[Document], dict]:
    """Busca as `k` receitas mais próximas e retorna (documentos, uso do embedding)."""
    vetor, tokens = _embedding_consulta(consulta)
    docs = obter_indice().similarity_search_by_vector(vetor, k=k)
    return docs, {"tokens_embedding": tokens, "custo_embedding_usd": custo_usd(MODELO_EMBEDDING, tokens)}
