# Skill: Event-Driven Architecture

> Referências: [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/), [Kafka Documentation](https://kafka.apache.org/documentation/), [CloudEvents Spec](https://cloudevents.io/), [Microservices.io Patterns](https://microservices.io/patterns/)

---

## Quando Usar Eventos

```
Comunicação SÍNCRONA (REST/gRPC):
  ✅ Precisa da resposta imediata para continuar
  ✅ Operação simples dentro do mesmo bounded context
  ✅ Leitura de dados (queries)

Comunicação ASSÍNCRONA (Eventos):
  ✅ Operação lenta (enviar email, gerar relatório, processar vídeo)
  ✅ Múltiplos consumidores precisam reagir ao mesmo evento
  ✅ Integração entre bounded contexts diferentes
  ✅ Aumentar resiliência (produtor e consumidor desacoplados)
  ✅ Auditoria e event sourcing
```

---

## Anatomia de um Evento

```json
{
  // CloudEvents envelope (padrão aberto)
  "specversion": "1.0",
  "type": "com.company.orders.order.confirmed",
  "source": "https://orders-service/orders",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "time": "2024-05-14T10:30:00Z",
  "datacontenttype": "application/json",

  // Payload do domínio
  "data": {
    "order_id": "order-uuid",
    "customer_id": "customer-uuid",
    "total_cents": 9990,
    "currency": "BRL",
    "confirmed_at": "2024-05-14T10:30:00Z"
  }
}
```

### Convenções de Nomeação

```
Formato:  {company}.{context}.{aggregate}.{past_verb}
Exemplos:
  com.company.orders.order.confirmed
  com.company.payments.payment.processed
  com.company.inventory.stock.reserved
  com.company.notifications.email.sent

Regras:
  ✅ Sempre no passado (algo que JÁ aconteceu)
  ✅ Hierarquia: empresa → contexto → aggregate → evento
  ❌ Nunca no imperativo (não "confirm.order", não "process.payment")
```

---

## Padrões de Integração

### 1. Transactional Outbox (Garantia de Entrega)

Problema: publicar evento e persistir dados atomicamente — sem perder eventos se o broker cair.

```
Fluxo:
  1. Use case: salva Order confirmada + insere evento na tabela outbox
     (uma transação de banco)
  2. Outbox Relay (processo separado): lê outbox e publica no broker
  3. Marca evento como processado na outbox

Nunca: publicar evento ANTES de commitar — pode publicar e falhar o commit.
Nunca: commitar ANTES de publicar — pode perder o evento se o processo morrer.
```

```python
# Use case: tudo na mesma transação de banco
def execute(self, request: ConfirmOrderRequest) -> ConfirmOrderResponse:
    with self._unit_of_work:
        order = self._order_repo.find_by_id(request.order_id)
        order.confirm()

        self._order_repo.save(order)

        # Salva na outbox — MESMA transação
        for event in order.pull_events():
            self._outbox_repo.save(OutboxEntry(
                aggregate_type="Order",
                aggregate_id=str(order.id),
                event_type=type(event).__name__,
                payload=event_to_dict(event),
            ))

        self._unit_of_work.commit()
    # Relay separado publicará no broker assincronamente
```

```sql
-- Tabela outbox (já definida em 08-data-architect)
CREATE TABLE domain_events_outbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(100) NOT NULL,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    retry_count INT NOT NULL DEFAULT 0
);
```

### 2. Consumer Groups (Kafka / RabbitMQ)

```
Tópico: orders.confirmed
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
  [Notification  [Inventory  [Analytics
   Consumer]      Consumer]   Consumer]
    group A         group B     group C

Cada consumer group recebe TODAS as mensagens do tópico.
Dentro do mesmo group, cada mensagem é processada por apenas 1 instância.
```

### 3. Saga Pattern (Transações Distribuídas)

Coordena operações em múltiplos serviços sem transação distribuída (XA).

**Choreography-based** (eventos → reações):
```
Order Service          Payment Service        Inventory Service
     │                       │                      │
  OrderConfirmed ──────────► │                      │
                        PaymentProcessed ──────────► │
                                               StockReserved
                                                     │
                                    (notifica Order Service)
```

**Orchestration-based** (saga coordinator):
```
Saga Coordinator:
  1. POST /payments  → success → continua
  2. POST /inventory → success → continua
  3. Se qualquer passo falha → compensa os anteriores (rollback manual)
```

---

## Garantias de Entrega

| Garantia | Significado | Consequência |
|----------|------------|--------------|
| **At-most-once** | Pode perder mensagens | Zero duplicatas, possível perda |
| **At-least-once** | Pode duplicar mensagens | Zero perda, possível duplicata |
| **Exactly-once** | Sem perda, sem duplicata | Mais complexo, menor throughput |

**Recomendação**: use **at-least-once** + **idempotência no consumidor** (mais simples que exactly-once).

```python
# Consumidor idempotente — verifica se já processou antes de agir
class OrderConfirmedHandler:
    def handle(self, event: OrderConfirmed) -> None:
        # Verifica idempotência pelo event_id
        if self._processed_events.exists(event.event_id):
            logger.info("Event already processed, skipping", event_id=event.event_id)
            return

        # Processa
        self._notification_service.send_confirmation(event.order_id)

        # Marca como processado
        self._processed_events.mark(event.event_id)
```

---

## Retry e Dead Letter Queue

```
Mensagem publicada
       │
  Consumidor tenta processar
       │
  Falha? ──────────────► Retry (com backoff exponencial)
                              │
                         3 tentativas falharam?
                              │
                         Dead Letter Queue (DLQ)
                              │
                         Alerta + investigação manual
```

```python
# Configuração de retry com backoff exponencial
retry_config = {
    "max_attempts": 3,
    "initial_delay_ms": 1000,
    "multiplier": 2.0,
    "max_delay_ms": 30000,
    # delays: 1s, 2s, 4s → DLQ
}
```

---

## Schema Registry e Evolução de Schema

Para evitar quebrar consumidores quando o schema do evento muda:

```
Regras de compatibilidade (Avro / Protobuf / JSON Schema):

BACKWARD compatible (consumidores antigos leem novos eventos):
  ✅ Adicionar campos opcionais com default
  ❌ Remover campos obrigatórios
  ❌ Renomear campos

FORWARD compatible (consumidores novos leem eventos antigos):
  ✅ Remover campos opcionais
  ❌ Adicionar campos obrigatórios sem default

FULL compatible: ambos — mais restritivo, mais seguro
```

---

## Configuração Kafka (Python)

```python
# Producer
from confluent_kafka import Producer

producer = Producer({
    "bootstrap.servers": "localhost:9092",
    "acks": "all",                    # Garantia de durabilidade
    "retries": 3,
    "retry.backoff.ms": 1000,
    "compression.type": "lz4",
})

def publish_event(topic: str, event: DomainEvent) -> None:
    producer.produce(
        topic=topic,
        key=str(event.aggregate_id).encode(),  # Garante ordering por aggregate
        value=json.dumps(asdict(event)).encode(),
        headers={"event-type": type(event).__name__},
    )
    producer.flush(timeout=5)

# Consumer
from confluent_kafka import Consumer

consumer = Consumer({
    "bootstrap.servers": "localhost:9092",
    "group.id": "notification-service",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,      # Commit manual após processamento
})

consumer.subscribe(["orders.confirmed"])

while True:
    msg = consumer.poll(timeout=1.0)
    if msg is None:
        continue
    try:
        event = parse_event(msg.value())
        handler.handle(event)
        consumer.commit(msg)           # Commit só após sucesso
    except Exception as e:
        logger.error("Handler failed, sending to DLQ", error=str(e))
        publish_to_dlq(msg)
        consumer.commit(msg)
```

---

## Checklist de Event-Driven

- [ ] Eventos nomeados no passado e seguindo hierarquia de namespace
- [ ] Envelope CloudEvents em todos os eventos publicados
- [ ] Transactional Outbox para garantia de entrega
- [ ] Consumidores idempotentes (verificam event_id antes de processar)
- [ ] Retry com backoff exponencial + Dead Letter Queue configurada
- [ ] Schema de evento documentado e versionado
- [ ] Consumer groups separados por bounded context consumidor
- [ ] Alertas para DLQ não vazia
- [ ] Saga pattern documentado para fluxos multi-serviço

---

## Instruções de Uso

```
@workspace #file:.github/copilot/skills/event-driven.md
#file:.github/copilot/templates/hexagonal/domain/domain-event.md
#file:src/domain/events/[event].py

Implemente o fluxo de publicação do evento [EventName].
Broker: [Kafka | RabbitMQ | Redis Streams | SQS]
Garantia necessária: [at-least-once | exactly-once]
Consumidores: [lista de serviços que reagem]
```
