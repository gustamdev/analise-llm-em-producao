from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=None)
def obter_llm(modelo: str, temperatura: float = 0.0) -> BaseChatModel:
    """Retorna o modelo (OpenAI por padrão), reaproveitando a instância."""
    return init_chat_model(modelo, model_provider=None if ":" in modelo else "openai",
                           temperature=temperatura, max_retries=2, timeout=60)
