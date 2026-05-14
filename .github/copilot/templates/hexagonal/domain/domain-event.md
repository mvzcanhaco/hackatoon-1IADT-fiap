# Template: Domain Event

## Quando Usar

Domain Events representam **algo que aconteceu** no domínio — um fato imutável do passado.

Use Domain Events quando:
- Uma mudança de estado numa entidade é relevante para outros contextos
- Outros Bounded Contexts precisam reagir a algo que aconteceu
- Você quer desacoplar o que aconteceu de quem reage a isso

**Regras obrigatórias**:
- Imutável (`frozen=True` / `readonly` / `data class`)
- Nomeado no **passado do indicativo**: `OrderConfirmed`, `PaymentProcessed`, `VideoAnalyzed`
- Gerado pela entidade ou agregado que sofreu a mudança
- Contém apenas dados necessários para os handlers reagirem

---

## Template: Domain Event Base

```python
# src/[context]/domain/events/base.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class DomainEvent:
    """
    Classe base para todos os Domain Events.

    Todo evento é imutável e carrega um ID único e timestamp de ocorrência.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

---

## Template: Domain Event Concreto

```python
# src/[context]/domain/events/[entity]_[past_verb].py
from __future__ import annotations

from dataclasses import dataclass
from .base import DomainEvent
from ..value_objects.[id_vo] import [EntityId]


@dataclass(frozen=True)
class [EntityName][PastVerb](DomainEvent):
    """
    Evento: [EntityName] foi [past_verb_description].

    Disparado quando: [descreva exatamente quando este evento ocorre]
    Produzido por: [EntityName].[method_name]()
    Consumido por: [liste os handlers ou bounded contexts que reagem]
    """
    [entity]_id: [EntityId]
    [campo_relevante_1]: [tipo]
    [campo_relevante_2]: [tipo]

    # Inclua apenas dados necessários para os handlers
    # Não inclua objetos mutáveis ou referências a entidades
```

### Exemplos concretos

```python
# Evento de confirmação de pedido
@dataclass(frozen=True)
class OrderConfirmed(DomainEvent):
    """Pedido foi confirmado e está pronto para processamento."""
    order_id: OrderId
    customer_id: CustomerId
    total_amount: Decimal
    item_count: int


# Evento de vídeo analisado
@dataclass(frozen=True)
class VideoAnalyzed(DomainEvent):
    """Vídeo foi processado e detecções foram geradas."""
    video_id: VideoId
    detection_count: int
    has_threats: bool
    processing_duration_ms: int


# Evento de pagamento processado
@dataclass(frozen=True)
class PaymentProcessed(DomainEvent):
    """Pagamento foi processado com sucesso."""
    order_id: OrderId
    payment_id: PaymentId
    amount: Decimal
    method: str
```

---

## Integração com Aggregate Root

Entities/Aggregates coletam eventos internamente e os retornam na operação:

```python
# src/[context]/domain/entities/[entity].py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from ..events.base import DomainEvent
from ..events.[entity]_[past_verb] import [EntityName][PastVerb]


@dataclass
class [EntityName]:
    """Aggregate root que produz domain events."""
    id: [EntityId]
    _events: List[DomainEvent] = field(default_factory=list, init=False, repr=False)

    def [action_method](self, [params]) -> None:
        """
        Executa [ação] e registra o evento correspondente.

        Raises:
            [DomainException]: quando [condição inválida]
        """
        # 1. Validar invariantes
        if [condição_inválida]:
            raise [DomainException]("...")

        # 2. Mudar estado
        self.[campo] = [novo_valor]

        # 3. Registrar evento (não publicar ainda)
        self._events.append(
            [EntityName][PastVerb](
                [entity]_id=self.id,
                [campo_relevante]=self.[campo],
            )
        )

    def pull_events(self) -> List[DomainEvent]:
        """
        Retorna e limpa os eventos acumulados.

        Chamado pelo Use Case após persistir a entidade.
        """
        events = list(self._events)
        self._events.clear()
        return events
```

---

## Uso no Use Case

```python
# src/[context]/application/use_cases/[use_case].py
@dataclass
class [UseCaseName]:
    repository: [EntityRepository]
    event_publisher: EventPublisher  # port secundário

    def execute(self, request: [RequestDTO]) -> [ResponseDTO]:
        # 1. Buscar entidade
        entity = self.repository.find_by_id(request.[id])

        # 2. Executar ação de domínio (acumula eventos internamente)
        entity.[action_method](request.[param])

        # 3. Persistir (antes de publicar eventos)
        self.repository.save(entity)

        # 4. Coletar e publicar eventos
        events = entity.pull_events()
        for event in events:
            self.event_publisher.publish(event)

        return [ResponseDTO](...)
```

---

## Port: Event Publisher

```python
# src/[context]/domain/ports/event_publisher.py
from abc import ABC, abstractmethod
from ..events.base import DomainEvent


class EventPublisher(ABC):
    """Port para publicação de Domain Events."""

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        """Publica um evento para os handlers registrados."""
        ...

    @abstractmethod
    def publish_all(self, events: list[DomainEvent]) -> None:
        """Publica múltiplos eventos em ordem."""
        ...
```

---

## Checklist

- [ ] Evento nomeado no passado do indicativo (`[Entity][PastVerb]`)
- [ ] `frozen=True` — imutável após criação
- [ ] Herda de `DomainEvent` (tem `event_id` e `occurred_at`)
- [ ] Gerado **dentro** da entidade/agregado no método que muda estado
- [ ] Use Case chama `pull_events()` **após** persistir
- [ ] Publicação acontece **depois** de `repository.save()`
- [ ] Contém apenas dados primitivos ou Value Objects imutáveis

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/domain/domain-event.md
#file:src/[context]/domain/entities/[entity].py
#file:.github/copilot/skills/domain-driven-design.md

Crie o Domain Event [EventName] para quando [entidade] [ação].
O evento deve conter: [lista de dados relevantes]
```
