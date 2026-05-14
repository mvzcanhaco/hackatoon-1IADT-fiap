# Template: Event Publisher (Infrastructure Adapter)

> Referência: [.github/copilot/skills/event-driven.md], [CloudEvents Spec](https://cloudevents.io/), [Transactional Outbox Pattern](https://microservices.io/patterns/data/transactional-outbox.html)

---

## Arquitetura do Publisher

```
Domain Layer:
  EventPublisher (Port/ABC) ← interface pura, sem dependência de infra

Infrastructure Layer:
  KafkaEventPublisher   → publica diretamente no broker (at-most-once)
  OutboxEventPublisher  → persiste na tabela outbox (at-least-once ✅ recomendado)
  InMemoryEventPublisher → para testes

Application Layer (Use Case):
  order.confirm()
  order_repo.save(order)
  events = order.pull_events()
  for event in events:
      publisher.publish(event)       ← via Port — não conhece a implementação
```

---

## Port: EventPublisher

```python
# src/domain/ports/event_publisher.py
from abc import ABC, abstractmethod
from src.domain.events.base import DomainEvent


class EventPublisher(ABC):
    """
    Secondary Port para publicação de domain events.
    Implementações: Kafka, RabbitMQ, Outbox, InMemory (testes).
    """

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        """Publica um único evento de domínio."""
        ...

    @abstractmethod
    def publish_all(self, events: list[DomainEvent]) -> None:
        """Publica múltiplos eventos atomicamente (se suportado pelo broker)."""
        ...
```

---

## Adapter 1: Transactional Outbox (recomendado para at-least-once)

```python
# src/infrastructure/messaging/outbox_event_publisher.py
import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.domain.events.base import DomainEvent
from src.domain.ports.event_publisher import EventPublisher

logger = logging.getLogger(__name__)


class OutboxEventPublisher(EventPublisher):
    """
    Persiste eventos na tabela outbox dentro da mesma transação de banco.
    Um processo separado (OutboxRelay) lê a outbox e publica no broker.
    Garante at-least-once delivery sem Two-Phase Commit.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def publish(self, event: DomainEvent) -> None:
        self.publish_all([event])

    def publish_all(self, events: list[DomainEvent]) -> None:
        for event in events:
            self._session.execute(
                """
                INSERT INTO domain_events_outbox
                    (id, aggregate_type, aggregate_id, event_type, payload, created_at)
                VALUES
                    (:id, :aggregate_type, :aggregate_id, :event_type, :payload, :created_at)
                """,
                {
                    "id": str(uuid.uuid4()),
                    "aggregate_type": event.__class__.__module__.split(".")[-2],
                    "aggregate_id": event.aggregate_id,
                    "event_type": type(event).__name__,
                    "payload": json.dumps(event.to_dict()),
                    "created_at": datetime.now(timezone.utc),
                },
            )
            logger.debug(
                "Event queued to outbox",
                extra={"event_type": type(event).__name__, "aggregate_id": event.aggregate_id},
            )
```

---

## Adapter 2: Kafka Publisher (at-least-once direto)

```python
# src/infrastructure/messaging/kafka_event_publisher.py
import json
import logging
from dataclasses import asdict

from confluent_kafka import Producer, KafkaException

from src.domain.events.base import DomainEvent
from src.domain.ports.event_publisher import EventPublisher

logger = logging.getLogger(__name__)

TOPIC_MAP: dict[str, str] = {
    "OrderConfirmed": "orders.confirmed",
    "OrderCancelled": "orders.cancelled",
    "PaymentProcessed": "payments.processed",
    # Adicione novos tipos aqui
}


class KafkaEventPublisher(EventPublisher):
    """
    Publica domain events diretamente no Kafka.
    Usa envelope CloudEvents para compatibilidade.
    Adequado quando a outbox não é necessária (eventos não-críticos).
    """

    def __init__(self, producer: Producer) -> None:
        self._producer = producer

    def publish(self, event: DomainEvent) -> None:
        self.publish_all([event])

    def publish_all(self, events: list[DomainEvent]) -> None:
        for event in events:
            topic = self._resolve_topic(event)
            envelope = self._to_cloudevent(event)
            try:
                self._producer.produce(
                    topic=topic,
                    key=event.aggregate_id.encode(),
                    value=json.dumps(envelope).encode(),
                    headers={"event-type": type(event).__name__},
                    on_delivery=self._delivery_report,
                )
            except KafkaException as exc:
                logger.error(
                    "Kafka publish failed",
                    extra={"event_type": type(event).__name__, "error": str(exc)},
                )
                raise
        self._producer.flush(timeout=5)

    def _resolve_topic(self, event: DomainEvent) -> str:
        event_type = type(event).__name__
        topic = TOPIC_MAP.get(event_type)
        if not topic:
            raise ValueError(f"No topic mapped for event type: {event_type}")
        return topic

    @staticmethod
    def _to_cloudevent(event: DomainEvent) -> dict:
        return {
            "specversion": "1.0",
            "type": f"com.company.{type(event).__name__.lower()}",
            "source": f"/{event.__class__.__module__.replace('.', '/')}",
            "id": event.event_id,
            "time": event.occurred_at.isoformat(),
            "datacontenttype": "application/json",
            "data": event.to_dict(),
        }

    @staticmethod
    def _delivery_report(err, msg) -> None:
        if err:
            logger.error("Kafka delivery failed", extra={"error": str(err), "topic": msg.topic()})
        else:
            logger.debug("Kafka delivery confirmed", extra={"topic": msg.topic(), "offset": msg.offset()})
```

---

## Outbox Relay (processo separado)

```python
# src/infrastructure/messaging/outbox_relay.py
import json
import logging
import time

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.infrastructure.messaging.kafka_event_publisher import KafkaEventPublisher

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 1
BATCH_SIZE = 100


class OutboxRelay:
    """
    Processo separado que lê a outbox e publica no broker.
    Roda em loop contínuo (worker/cron/sidecar).
    Garante at-least-once: se o broker cair, a outbox preserva os eventos.
    """

    def __init__(self, session: Session, publisher: KafkaEventPublisher) -> None:
        self._session = session
        self._publisher = publisher

    def run(self) -> None:
        logger.info("OutboxRelay started")
        while True:
            try:
                self._process_batch()
            except Exception as exc:
                logger.error("OutboxRelay batch failed", extra={"error": str(exc)})
            time.sleep(POLL_INTERVAL_SECONDS)

    def _process_batch(self) -> None:
        rows = self._session.execute(
            text("""
                SELECT id, event_type, payload
                FROM domain_events_outbox
                WHERE processed_at IS NULL
                ORDER BY created_at ASC
                LIMIT :limit
                FOR UPDATE SKIP LOCKED
            """),
            {"limit": BATCH_SIZE},
        ).mappings().all()

        if not rows:
            return

        for row in rows:
            try:
                event_dict = json.loads(row["payload"])
                self._publisher.publish_raw(
                    event_type=row["event_type"],
                    payload=event_dict,
                )
                self._session.execute(
                    text("UPDATE domain_events_outbox SET processed_at = NOW() WHERE id = :id"),
                    {"id": row["id"]},
                )
                logger.info(
                    "Outbox event published",
                    extra={"event_type": row["event_type"], "outbox_id": row["id"]},
                )
            except Exception as exc:
                self._session.execute(
                    text("UPDATE domain_events_outbox SET retry_count = retry_count + 1 WHERE id = :id"),
                    {"id": row["id"]},
                )
                logger.error(
                    "Outbox event publish failed",
                    extra={"event_type": row["event_type"], "error": str(exc)},
                )

        self._session.commit()
```

---

## Adapter 3: InMemory (para testes)

```python
# src/infrastructure/messaging/in_memory_event_publisher.py
from src.domain.events.base import DomainEvent
from src.domain.ports.event_publisher import EventPublisher


class InMemoryEventPublisher(EventPublisher):
    """
    Publisher em memória para testes unitários e de integração.
    Permite assertar quais eventos foram publicados.
    """

    def __init__(self) -> None:
        self.published: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.published.append(event)

    def publish_all(self, events: list[DomainEvent]) -> None:
        self.published.extend(events)

    def reset(self) -> None:
        self.published.clear()

    def published_of_type(self, event_type: type) -> list[DomainEvent]:
        return [e for e in self.published if isinstance(e, event_type)]
```

---

## Uso em Testes

```python
# tests/unit/application/test_confirm_order_use_case.py
from src.infrastructure.messaging.in_memory_event_publisher import InMemoryEventPublisher
from src.domain.events.order_confirmed import OrderConfirmed


def test_confirm_order_publishes_event(mock_order_repo, sample_order):
    publisher = InMemoryEventPublisher()
    use_case = ConfirmOrderUseCase(order_repo=mock_order_repo, event_publisher=publisher)

    use_case.execute(ConfirmOrderRequest(order_id=str(sample_order.id)))

    events = publisher.published_of_type(OrderConfirmed)
    assert len(events) == 1
    assert events[0].aggregate_id == str(sample_order.id)
```

---

## Checklist do Event Publisher

```
Design:
  [ ] Port (ABC) na camada de domínio — sem referência a Kafka/RabbitMQ
  [ ] Adaptadores concretos apenas na camada de infraestrutura
  [ ] InMemoryEventPublisher disponível para testes
  [ ] TOPIC_MAP centralizado — nunca hardcoded nos publishers

Entrega:
  [ ] Outbox pattern para eventos de negócio críticos (at-least-once)
  [ ] Kafka Publisher direto apenas para eventos não-críticos (métricas, logs)
  [ ] OutboxRelay com FOR UPDATE SKIP LOCKED (evita duplicação entre instâncias)

Formato:
  [ ] CloudEvents envelope em todos os eventos publicados
  [ ] event_id como UUID único por evento (idempotência no consumidor)
  [ ] aggregate_id como partition key no Kafka (ordering por aggregate)

Monitoramento:
  [ ] Log de delivery confirmation no Kafka (on_delivery callback)
  [ ] Log de retry no OutboxRelay com retry_count
  [ ] Alerta se outbox tiver eventos não processados há mais de X minutos
```

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/infrastructure/event-publisher.md
#file:.github/copilot/skills/event-driven.md
#file:.github/copilot/templates/hexagonal/domain/domain-event.md
#file:src/domain/events/[event].py

Implemente o publisher para o evento [EventName].
Broker: [Kafka | RabbitMQ | Redis Streams | SQS]
Garantia: [at-least-once via Outbox | at-most-once direto]
Tópico destino: [nome do tópico]
```
