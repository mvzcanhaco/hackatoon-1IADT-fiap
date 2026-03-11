# Template: Entity (Entidade de Domínio)

## Quando Usar

Use este template quando o objeto:
- Tem identidade única que persiste ao longo do tempo
- Pode ter seus atributos alterados mantendo a mesma identidade
- Encapsula regras de negócio e invariantes
- Produz eventos de domínio quando muda de estado

---

## Template Completo

```python
# src/[context]/domain/entities/[entity_name].py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from ..[value_objects].[id_vo] import [EntityId]
from ..[value_objects].[status_vo] import [StatusVO]
from ..[events].[event_module] import [EntityCreatedEvent], [StateChangedEvent]
from ..[exceptions].[context]_exceptions import (
    [InvalidStateTransitionError],
    [BusinessRuleViolationError],
)


@dataclass
class [EntityName]:
    """
    Entity: [EntityName]

    Representa [descreva o que esta entidade representa no domínio negócio].

    Responsabilidades:
    - [responsabilidade 1]
    - [responsabilidade 2]

    Invariantes (nunca violadas):
    - [invariante 1: ex: status não pode regredir]
    - [invariante 2: ex: total nunca negativo]
    - [invariante 3: ex: deve ter ao menos um item]
    """

    # ─── Identidade ──────────────────────────────────────────────────────────
    id: [EntityId]

    # ─── Atributos ───────────────────────────────────────────────────────────
    [campo_obrigatorio]: [Tipo]
    [campo_opcional]: Optional[[Tipo]] = None

    # ─── Estado interno (encapsulado) ─────────────────────────────────────────
    _[colecao_interna]: List[[TipoMembro]] = field(default_factory=list, repr=False)
    _status: [StatusVO] = field(default=[StatusVO].INITIAL, repr=True)
    _created_at: datetime = field(default_factory=datetime.utcnow, repr=False)
    _updated_at: Optional[datetime] = field(default=None, repr=False)

    # ─── Propriedades (leitura segura do estado interno) ───────────────────────

    @property
    def status(self) -> [StatusVO]:
        return self._status

    @property
    def [colecao_interna](self) -> tuple:
        """Retorna tupla imutável — externos não modificam a coleção."""
        return tuple(self._[colecao_interna])

    @property
    def created_at(self) -> datetime:
        return self._created_at

    # ─── Comportamentos de Negócio (Comandos) ────────────────────────────────

    def [acao_principal](
        self,
        [param]: [Tipo],
        [param2]: Optional[[Tipo]] = None,
    ) -> [StateChangedEvent]:
        """
        [Descreva o comportamento em linguagem de domínio].

        Pré-condições:
        - [condição que deve ser verdadeira para executar]

        Pós-condições:
        - [o que muda após execução]

        Args:
            [param]: [descrição]

        Returns:
            [StateChangedEvent]: evento emitido para notificar handlers

        Raises:
            [InvalidStateTransitionError]: quando [condição de erro]
            [BusinessRuleViolationError]: quando [regra de negócio violada]
        """
        self._validate_can_[acao]()
        # lógica de negócio
        self._status = [StatusVO].NEW_STATE
        self._updated_at = datetime.utcnow()
        return [StateChangedEvent](entity_id=self.id, [dados_relevantes]=...)

    def add_[membro](
        self,
        [param]: [Tipo],
    ) -> None:
        """
        Adiciona [membro] à coleção.

        Invariante: [condição que garante consistência]

        Raises:
            [InvalidStateTransitionError]: quando entidade não permite adição
        """
        if self._status not in [StatusVO].editable_states():
            raise [InvalidStateTransitionError](
                entity_id=self.id,
                current_status=self._status,
                attempted_action="add_[membro]",
            )
        self._[colecao_interna].append([TipoMembro]([param]))
        self._updated_at = datetime.utcnow()

    # ─── Consultas (Queries) ──────────────────────────────────────────────────

    def [calculo_de_negocio](self) -> [ReturnType]:
        """[Descreva o que calcula — sem efeitos colaterais]."""
        return sum(item.[campo] for item in self._[colecao_interna])

    def has_[condição](self) -> bool:
        """Verifica se [condição de negócio]."""
        return bool(self._[colecao_interna])

    # ─── Validações Privadas ──────────────────────────────────────────────────

    def _validate_can_[acao](self) -> None:
        """Valida pré-condições para [ação]."""
        if self._status == [StatusVO].FINAL_STATE:
            raise [InvalidStateTransitionError](
                entity_id=self.id,
                current_status=self._status,
                attempted_action="[acao]",
            )

    # ─── Factory Methods ─────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        [param]: [Tipo],
        entity_id: Optional[[EntityId]] = None,
    ) -> tuple["[EntityName]", [EntityCreatedEvent]]:
        """
        Factory method para criar nova instância válida.

        Returns:
            Tupla de (entidade criada, evento de criação)
        """
        entity = cls(
            id=entity_id or [EntityId](str(uuid4())),
            [campo_obrigatorio]=[param],
        )
        event = [EntityCreatedEvent](entity_id=entity.id)
        return entity, event

    # ─── Representação ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"[EntityName](id={self.id}, status={self._status})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, [EntityName]):
            return False
        return self.id == other.id  # Igualdade por identidade, não por valor

    def __hash__(self) -> int:
        return hash(self.id)
```

---

## Checklist

- [ ] Identidade única (campo `id` com Value Object de ID)
- [ ] Estado interno encapsulado (campos privados com `_`)
- [ ] Propriedades de leitura retornam imutáveis (tuple, não list)
- [ ] Métodos de negócio validam pré-condições
- [ ] Exceções de domínio específicas (não `ValueError` genérico)
- [ ] Eventos de domínio retornados (não publicados diretamente)
- [ ] `__eq__` e `__hash__` baseados na identidade
- [ ] Zero imports de frameworks externos
- [ ] Testes unitários cobrindo todos os comportamentos

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/domain/entity.md
#file:.github/copilot/skills/domain-driven-design.md

Crie a entidade [NomeDaEntidade] com:
- Atributos: [lista]
- Comportamentos: [lista de ações de negócio]
- Invariantes: [lista das regras que nunca podem ser violadas]
- Eventos emitidos: [lista]
```
