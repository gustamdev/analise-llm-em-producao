"""Schemas Pydantic do projeto."""

from typing import Literal

from pydantic import BaseModel, Field

Complexidade = Literal["facil", "medio", "dificil"]


class ItemGeladeira(BaseModel):
    nome: str
    quantidade: float
    unidade: str
    categoria: str
    validade: str  # data ISO (AAAA-MM-DD)


class Receita(BaseModel):
    """Saída estruturada da tool `criar_receita` — o contrato que o frontend consome."""

    nome: str = Field(min_length=3, description="Nome do prato, em português")
    descricao: str = Field(min_length=10, description="Uma ou duas frases apresentando o prato")
    ingredientes: list[str] = Field(
        min_length=2, description="Ingredientes com quantidade, ex.: '200 g de tomate'"
    )
    modo_preparo: list[str] = Field(min_length=2, description="Passos numerados em ordem")
    tempo_preparo_min: int = Field(gt=0, le=600, description="Tempo total em minutos")
    porcoes: int = Field(gt=0, le=20)
    ingredientes_prioritarios_usados: list[str] = Field(
        default_factory=list,
        description="Quais ingredientes perto do vencimento foram aproveitados",
    )


class AvaliacaoJuiz(BaseModel):
    """Saída estruturada do LLM-as-judge. O raciocínio vem ANTES das notas de propósito:
    pedir a justificativa primeiro tende a deixar a nota mais consistente."""

    raciocinio: str = Field(description="Análise curta, critério por critério")
    uso_da_geladeira: int = Field(
        ge=1, le=5, description="5 = usa só itens da geladeira (+ básicos como sal, óleo, água)"
    )
    prioriza_vencimento: int = Field(
        ge=1, le=5, description="5 = aproveita os itens que vencem logo e NÃO usa itens vencidos"
    )
    coerencia_preparo: int = Field(
        ge=1, le=5, description="5 = passos claros, ordem lógica, tempos plausíveis"
    )
    atende_pedido: int = Field(ge=1, le=5, description="5 = responde exatamente ao que foi pedido")

    def nota(self) -> float:
        """Média dos critérios convertida para 0..1."""
        notas = [
            self.uso_da_geladeira,
            self.prioriza_vencimento,
            self.coerencia_preparo,
            self.atende_pedido,
        ]
        return round((sum(notas) / len(notas) - 1) / 4, 3)


class ClassificacaoComplexidade(BaseModel):
    """Saída do roteador: qual o nível de dificuldade do pedido."""

    motivo: str = Field(description="Justificativa em uma frase")
    nivel: Complexidade
