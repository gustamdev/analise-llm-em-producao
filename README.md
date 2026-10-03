# Análise de LLM em produção

Como avaliar qualidade, controlar custo e observar uma aplicação com LLM, usando como caso um agente que sugere receitas a partir de uma geladeira virtual.

**Stack:** Python · LangChain · LangGraph · LangSmith · OpenAI · Chroma · Pydantic · Streamlit

## O que o projeto mostra

- **Agente com tool calling (LangGraph):** 4 ferramentas com perfis de custo diferentes (código puro, RAG e LLM).
- **RAG com dados reais:** 2.000 receitas do Epicurious indexadas num Chroma local.
- **Saída validada com Pydantic:** a receita segue um contrato fixo, como o que um frontend consumiria.
- **Golden dataset congelado no LangSmith:** 16 casos em 3 níveis de dificuldade, com as ferramentas esperadas como gabarito.
- **Avaliação automática:** validação de schema, *tool selection* (exata e F1) e LLM-as-judge calibrado antes do uso.
- **Observabilidade:** custo, latência e volume por ferramenta, com dashboard em Streamlit.
- **Roteamento de modelos:** um classificador barato decide entre `gpt-4.1`, `gpt-4.1-mini` e `gpt-4.1-nano`.

## Resultados

| Ferramenta | Tipo | Latência | Custo por chamada |
|---|---|---|---|
| `check_geladeira` | código puro | ~0,02 ms | US$ 0 |
| `verificar_validade` | código puro | ~0,03 ms | US$ 0 |
| `buscar_receitas_similares` | RAG | ~0,5 s | ~US$ 0 (embedding) + tokens no prompt |
| `criar_receita` | LLM + Pydantic | ~4 s | ~US$ 0,002 |

**Teste de regressão de prompt** (`gpt-4.1-mini`, mesmo dataset): o prompt mais curto ficou 25% mais barato, mas a qualidade caiu.

| Métrica | Prompt v1 | Prompt v2 (enxuto) |
|---|---|---|
| Nota do juiz | 0,92 | 0,85 |
| Schema válido | 0,93 | 0,86 |
| Custo por pedido | US$ 0,0032 | US$ 0,0024 |
| Latência | 7,5 s | 6,1 s |

**Aprendizados**
- Quase todo o custo está nas etapas com LLM; o custo real do RAG é o contexto injetado, não o embedding.
- O juiz separou corretamente a receita boa (1,0) de versões corrompidas (0,50 a 0,75).
- A mesma configuração variou 0,10 entre duas execuções: decisões de modelo precisam de repetições.
- O LLM declarou ter usado ingredientes que não estavam na receita, por isso as métricas conferem o resultado real.

## Notebooks

1. [Setup e tools](notebooks/01_setup_e_tools.ipynb): ferramentas, índice RAG, trade-off de `k` e agente completo.
2. [Golden dataset](notebooks/02_golden_dataset.ipynb): criação, congelamento e publicação no LangSmith.
3. [Avaliação e LLM-as-judge](notebooks/03_avaliacao_llm_judge.ipynb): calibração do juiz, experimentos e detecção de regressão.
4. [Observabilidade e custos](notebooks/04_observabilidade_custos.ipynb): tráfego simulado e custo por ferramenta.
5. [Roteamento de modelos](notebooks/05_roteamento_modelos.ipynb): baseline no `gpt-4.1`, comparação com modelos mais baratos por nível de dificuldade e validação do roteador ponta a ponta.

Dados: [Hieu-Pham/kaggle_food_recipes](https://huggingface.co/datasets/Hieu-Pham/kaggle_food_recipes) (CC-BY-SA 3.0).
