# Skill: Arquitetura Hexagonal (Ports & Adapters)

## O que é

A Arquitetura Hexagonal (Alistair Cockburn, 2005) — também chamada Ports & Adapters — isola o núcleo da aplicação (domínio + casos de uso) de suas dependências externas (banco de dados, APIs, UI, frameworks).

**Regra de Ouro**: as dependências apontam para dentro. O domínio nunca conhece a infraestrutura.

```
┌─────────────────────────────────────────────────────────────┐
│                    MUNDO EXTERNO                            │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────────┐ │
│  │HTTP  │  │ CLI  │  │Kafka │  │Email │  │    Banco     │ │
│  │REST  │  │      │  │Event │  │      │  │    de Dados  │ │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──────┬───────┘ │
│     │         │          │          │              │         │
│  ┌──▼───────────────────────────────▼──────────────▼──────┐ │
│  │                   ADAPTERS                             │ │
│  │  (Controllers, CLI Handlers, Event Consumers,          │ │
│  │   Repository Implementations, Email Senders)           │ │
│  └──┬─────────────────────────────────────────────────┬───┘ │
│     │                                                 │     │
│  ┌──▼─────────────────────────────────────────────────▼───┐ │
│  │                   PORTS (Interfaces)                   │ │
│  │  (Repository Interfaces, Notification Interfaces,      │ │
│  │   Service Interfaces — definidas no domínio/app)       │ │
│  └──┬─────────────────────────────────────────────────┬───┘ │
│     │                                                 │     │
│  ┌──▼─────────────────────────────────────────────────▼───┐ │
│  │              NÚCLEO DA APLICAÇÃO                        │ │
│  │  ┌────────────────────┐  ┌──────────────────────────┐  │ │
│  │  │     DOMÍNIO        │  │   APLICAÇÃO (Use Cases)  │  │ │
│  │  │  Entities          │  │  ProcessVideoUseCase      │  │ │
│  │  │  Value Objects     │←─│  CreateOrderUseCase       │  │ │
│  │  │  Domain Services   │  │  AuthenticateUserUseCase  │  │ │
│  │  │  Domain Events     │  │                           │  │ │
│  │  │  Aggregate Roots   │  │  (orquestra o domínio)    │  │ │
│  │  └────────────────────┘  └──────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## As 4 Camadas em Detalhe

### 1. Domain (Domínio)

**O quê**: Regras de negócio puras. O coração do software.

**Contém**:
- `entities/` — objetos com identidade e comportamento
- `value_objects/` — objetos imutáveis descritos por seus valores
- `aggregates/` — clusters de objetos com consistência garantida
- `repositories/` — **interfaces** (ports) de persistência
- `services/` — lógica de domínio que não pertence a uma entidade
- `events/` — eventos que representam fatos do domínio
- `factories/` — criação de objetos complexos
- `exceptions/` — exceções de negócio

**Regras**:
- ❌ NUNCA importe de `infrastructure`, `presentation` ou frameworks
- ❌ NUNCA use SQLAlchemy, FastAPI, requests, etc.
- ✅ Apenas Python puro (dataclasses, ABC, typing)

```python
# ✅ Correto — domínio puro
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from .value_objects.money import Money

# ❌ Errado — framework no domínio
from sqlalchemy import Column, Integer  # NUNCA no domínio
from fastapi import HTTPException        # NUNCA no domínio
```

---

### 2. Application (Aplicação)

**O quê**: Orquestra o domínio. Define os casos de uso do sistema.

**Contém**:
- `use_cases/` — um arquivo por operação de negócio
- `dto/` — objetos de transferência (Request/Response)
- `ports/` — interfaces para serviços externos que o use case precisa

**Regras**:
- ❌ NUNCA contém lógica de negócio (isso é responsabilidade do domínio)
- ❌ NUNCA conhece HTTP, SQL, ou frameworks específicos
- ✅ Orquestra chamadas ao domínio e à infraestrutura (via interfaces)

```python
# ✅ Correto — use case orquestra domínio
class ProcessOrderUseCase:
    def __init__(self, order_repo: OrderRepository, payment_service: PaymentPort):
        self._order_repo = order_repo
        self._payment_service = payment_service

    def execute(self, request: ProcessOrderRequest) -> ProcessOrderResponse:
        order = self._order_repo.find_by_id(request.order_id)  # busca via port
        event = order.confirm()                                  # regra no domínio
        self._payment_service.charge(order.total)               # serviço via port
        self._order_repo.save(order)                            # persiste via port
        return ProcessOrderResponse(order_id=order.id)
```

---

### 3. Infrastructure (Infraestrutura)

**O quê**: Implementações concretas dos ports. Adaptadores para o mundo externo.

**Contém**:
- `repositories/` — implementações concretas com SQLAlchemy, MongoDB, etc.
- `adapters/` — clients de APIs externas, brokers de mensagem
- `services/` — implementações de serviços externos (email, SMS, notificações)
- `config/` — configurações, injeção de dependências

**Regras**:
- ✅ Pode importar frameworks (SQLAlchemy, httpx, boto3, etc.)
- ✅ Implementa as interfaces definidas no domínio/aplicação
- ❌ NUNCA contém lógica de negócio

```python
# ✅ Correto — implementação concreta do port
class SqlAlchemyOrderRepository(OrderRepository):  # implementa a interface do domínio
    def __init__(self, session: Session):
        self._session = session

    def find_by_id(self, order_id: OrderId) -> Optional[Order]:
        record = self._session.query(OrderModel).get(str(order_id))
        return self._to_domain(record) if record else None
```

---

### 4. Presentation (Apresentação)

**O quê**: Interface com o mundo externo que dispara use cases.

**Contém**:
- `http/` ou `web/` — controllers REST, GraphQL resolvers
- `cli/` — comandos de terminal
- `events/` — consumers de eventos (Kafka, RabbitMQ)
- `grpc/` — endpoints gRPC

**Regras**:
- ✅ Recebe input externo e chama use cases
- ✅ Converte erros de domínio em respostas adequadas (HTTP 4xx/5xx)
- ❌ NUNCA chama repositórios diretamente
- ❌ NUNCA contém lógica de negócio

---

## Fluxo de uma Requisição

```
HTTP Request
    ↓
[Controller] ← Presentation
    ↓ (chama)
[UseCase.execute(RequestDTO)] ← Application
    ↓ (usa)
[Entity.action()] ← Domain
    ↓ (via interface)
[Repository.save()] ← Domain (interface)
    ↓ (implementado por)
[SqlAlchemyRepository] ← Infrastructure
    ↓
Database
```

---

## Regra das Dependências — Quick Reference

| Camada | Pode importar de |
|--------|-----------------|
| Domain | Ninguém (apenas stdlib) |
| Application | Domain |
| Infrastructure | Domain, Application |
| Presentation | Application (use cases e DTOs) |

---

## Anti-patterns Comuns

### ❌ Fat Controller
```python
# Ruim — lógica de negócio no controller
@router.post("/orders")
async def create_order(payload: CreateOrderPayload, db: Session = Depends(get_db)):
    # ERRADO: lógica de negócio na presentation
    if payload.quantity <= 0:
        raise HTTPException(400, "Quantity must be positive")
    order = Order(customer_id=payload.customer_id, quantity=payload.quantity)
    db.add(order)  # ERRADO: acesso direto ao banco
    db.commit()
    return {"id": order.id}
```

### ✅ Correto
```python
@router.post("/orders", status_code=201)
async def create_order(
    payload: CreateOrderPayload,
    use_case: CreateOrderUseCase = Depends(get_create_order_use_case),
) -> CreateOrderResponseSchema:
    try:
        response = use_case.execute(CreateOrderRequest(
            customer_id=payload.customer_id,
            quantity=payload.quantity,
        ))
        return CreateOrderResponseSchema(id=response.order_id)
    except InvalidQuantityError as e:
        raise HTTPException(status_code=422, detail=str(e))
```

---

## Quando Usar Este Skill

- Decidir onde colocar um novo arquivo ou classe
- Revisar se uma classe está na camada correta
- Entender o fluxo de uma feature de ponta a ponta
- Debugar violações de arquitetura em code review
