# Template: Aggregate Root

> Referência: [DDD — Eric Evans](https://www.domainlanguage.com/ddd/), [Implementing DDD — Vernon](https://vaughnvernon.co/?page_id=168), [.github/copilot/skills/domain-driven-design.md]

---

## O que é um Aggregate Root

O Aggregate Root é a **única entrada** para um agregado de objetos de domínio. Todo acesso externo passa pelo root — nunca diretamente pelos filhos.

```
Regras do Aggregate:
  1. A identidade do root é global; identidades dos filhos são locais ao aggregate
  2. Objetos externos referem-se ao aggregate apenas pelo root (via ID)
  3. O root garante invariantes: nenhum filho pode violar a integridade do aggregate
  4. Uma transação = um aggregate (evitar transações que cruzam fronteiras de aggregate)
  5. Comunicação entre aggregates: via domain events (assíncrono) ou IDs (síncrono)
```

---

## Template Base

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.domain.events.base import DomainEvent

if TYPE_CHECKING:
    pass


@dataclass
class [AggregateRoot]:
    """
    Aggregate root de [Bounded Context].

    Invariantes:
      - [invariante 1: descreva a regra de negócio]
      - [invariante 2]
    """

    id: [AggregateId]
    # --- Campos de estado ---
    [campo]: [Tipo]
    [status]: [StatusEnum]

    # --- Metadados ---
    created_at: datetime
    updated_at: datetime

    # --- Eventos (não persistido) ---
    _events: list[DomainEvent] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, [params]) -> [AggregateRoot]:
        """Cria nova instância validando invariantes iniciais."""
        # Validações de criação
        if not [condição]:
            raise [ValidationError]("[mensagem com contexto]")

        now = datetime.now(timezone.utc)
        instance = cls(
            id=[AggregateId](uuid.uuid4()),
            [campo]=[valor],
            [status]=[StatusEnum].[INITIAL],
            created_at=now,
            updated_at=now,
        )

        # Publica evento de criação
        instance._events.append(
            [AggregateCreated](
                aggregate_id=str(instance.id),
                [campos_do_evento]=[valores],
            )
        )

        return instance

    @classmethod
    def reconstitute(cls, **kwargs) -> [AggregateRoot]:
        """
        Reconstrói aggregate a partir do estado persistido.
        Chamado pelo repositório — nunca pela aplicação diretamente.
        NÃO publica eventos (estado já consolidado).
        """
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Comandos (state-changing operations)
    # ------------------------------------------------------------------

    def [acao](self, [params]) -> None:
        """
        [Descrição do comportamento].
        Pré-condição: [estado esperado antes]
        Pós-condição: [estado esperado depois]
        """
        self._assert_can_[acao]()

        # Alterar estado
        self.[campo] = [novo_valor]
        self.[status] = [StatusEnum].[NOVO_STATUS]
        self.updated_at = datetime.now(timezone.utc)

        # Registrar evento
        self._events.append(
            [AggregateActionCompleted](
                aggregate_id=str(self.id),
                [campos_do_evento]=[valores],
            )
        )

    # ------------------------------------------------------------------
    # Queries (read-only, sem efeitos colaterais)
    # ------------------------------------------------------------------

    @property
    def is_[estado](self) -> bool:
        return self.[status] == [StatusEnum].[ESTADO]

    def [calcular_algo](self) -> [Tipo]:
        """Derivação calculada a partir do estado — sem alterar nada."""
        return [cálculo]

    # ------------------------------------------------------------------
    # Invariantes (privados)
    # ------------------------------------------------------------------

    def _assert_can_[acao](self) -> None:
        if self.[status] not in {[StatusEnum].[ESTADO_PERMITIDO_1], [StatusEnum].[ESTADO_PERMITIDO_2]}:
            raise [InvalidStateTransitionError](
                entity=type(self).__name__,
                entity_id=str(self.id),
                current=[self.[status].value],
                target="[acao]",
            )

    # ------------------------------------------------------------------
    # Domain Events
    # ------------------------------------------------------------------

    def pull_events(self) -> list[DomainEvent]:
        """Retorna e limpa os eventos pendentes. Chamado pelo Use Case após save()."""
        events, self._events = self._events, []
        return events
```

---

## Exemplo Concreto: Order Aggregate

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from src.domain.value_objects.order_id import OrderId
from src.domain.value_objects.customer_id import CustomerId
from src.domain.value_objects.money import Money
from src.domain.entities.order_item import OrderItem
from src.domain.enums.order_status import OrderStatus
from src.domain.events.order_confirmed import OrderConfirmed
from src.domain.events.order_cancelled import OrderCancelled
from src.domain.exceptions import (
    BusinessRuleViolationError,
    InvalidStateTransitionError,
    InvariantViolationError,
)
from src.domain.events.base import DomainEvent


@dataclass
class Order:
    """
    Aggregate root de Orders.

    Invariantes:
      - Um pedido confirmado não pode ser editado
      - Um pedido cancelado não pode ser confirmado
      - O total deve refletir a soma dos itens
    """

    id: OrderId
    customer_id: CustomerId
    items: list[OrderItem]
    status: OrderStatus
    created_at: datetime
    updated_at: datetime

    _events: list[DomainEvent] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, customer_id: CustomerId, items: list[OrderItem]) -> Order:
        if not items:
            raise InvariantViolationError(
                entity="Order",
                invariant="items_not_empty",
                detail="An order must have at least one item",
            )

        now = datetime.now(timezone.utc)
        order = cls(
            id=OrderId(uuid.uuid4()),
            customer_id=customer_id,
            items=list(items),
            status=OrderStatus.DRAFT,
            created_at=now,
            updated_at=now,
        )
        return order

    @classmethod
    def reconstitute(
        cls,
        id: OrderId,
        customer_id: CustomerId,
        items: list[OrderItem],
        status: OrderStatus,
        created_at: datetime,
        updated_at: datetime,
    ) -> Order:
        return cls(
            id=id,
            customer_id=customer_id,
            items=items,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
        )

    # ------------------------------------------------------------------
    # Comandos
    # ------------------------------------------------------------------

    def confirm(self) -> None:
        self._assert_can_confirm()
        self.status = OrderStatus.CONFIRMED
        self.updated_at = datetime.now(timezone.utc)
        self._events.append(
            OrderConfirmed(
                aggregate_id=str(self.id),
                customer_id=str(self.customer_id),
                total_cents=self.total.cents,
            )
        )

    def cancel(self, reason: str) -> None:
        self._assert_can_cancel()
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)
        self._events.append(
            OrderCancelled(
                aggregate_id=str(self.id),
                reason=reason,
            )
        )

    def add_item(self, item: OrderItem) -> None:
        self._assert_is_editable()
        self.items.append(item)
        self.updated_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total(self) -> Money:
        return sum((item.subtotal for item in self.items), Money.zero())

    @property
    def is_confirmed(self) -> bool:
        return self.status == OrderStatus.CONFIRMED

    @property
    def item_count(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    # Invariantes
    # ------------------------------------------------------------------

    def _assert_can_confirm(self) -> None:
        if self.status != OrderStatus.DRAFT:
            raise InvalidStateTransitionError(
                entity="Order",
                entity_id=str(self.id),
                current=self.status.value,
                target="confirmed",
            )

    def _assert_can_cancel(self) -> None:
        if self.status == OrderStatus.CANCELLED:
            raise InvalidStateTransitionError(
                entity="Order",
                entity_id=str(self.id),
                current=self.status.value,
                target="cancelled",
            )

    def _assert_is_editable(self) -> None:
        if self.status != OrderStatus.DRAFT:
            raise BusinessRuleViolationError(
                rule="order_editable_only_in_draft",
                context={"order_id": str(self.id), "status": self.status.value},
            )

    # ------------------------------------------------------------------
    # Domain Events
    # ------------------------------------------------------------------

    def pull_events(self) -> list[DomainEvent]:
        events, self._events = self._events, []
        return events
```

---

## Checklist do Aggregate Root

```
Design:
  [ ] Bounded context identificado — aggregate não cruza contextos
  [ ] Invariantes documentadas no docstring da classe
  [ ] Filhos do aggregate referenciados por ID (não por objeto) quando cross-aggregate
  [ ] Nenhuma dependência de infraestrutura (ORM, HTTP, Redis) no aggregate

Comandos:
  [ ] Todos os métodos que alteram estado são verbos no imperativo (confirm, cancel, add_item)
  [ ] Toda transição de estado valida invariantes antes de aplicar
  [ ] Timestamp updated_at atualizado em toda mutação
  [ ] Evento de domínio registrado em _events para cada mudança significativa

Queries:
  [ ] Propriedades derivadas são @property — nunca alteram estado
  [ ] Métodos is_* e has_* retornam bool, sem efeitos colaterais

Factory:
  [ ] Método create() para instâncias novas (publica evento de criação)
  [ ] Método reconstitute() para restaurar do banco (sem eventos)
  [ ] Validação de invariantes na criação

Domain Events:
  [ ] pull_events() limpa a lista após retornar (evitar double-publish)
  [ ] Use Case chama pull_events() APÓS save() bem-sucedido
```

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/domain/aggregate-root.md
#file:.github/copilot/templates/hexagonal/domain/domain-event.md
#file:.github/copilot/templates/hexagonal/domain/exceptions.md
#file:.github/copilot/agents/03-domain-modeler.md

Implemente o aggregate root [AggregateName] com as seguintes invariantes:
  - [invariante 1]
  - [invariante 2]
Transições de estado: [DRAFT → ACTIVE → COMPLETED | CANCELLED]
Eventos que deve emitir: [EventA, EventB]
```
