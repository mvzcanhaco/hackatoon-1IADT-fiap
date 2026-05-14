# Template: Port Interface

## Quando Usar

Ports são **contratos (interfaces abstratas)** que definem o que o domínio/aplicação precisa do mundo externo — sem saber como será implementado.

Existem dois tipos:

| Tipo | Direção | Quem implementa | Exemplos |
|------|---------|-----------------|---------|
| **Primary Port** (Driving) | Usuário → Aplicação | Use Cases | `ProcessVideoUseCase`, `CreateOrderUseCase` |
| **Secondary Port** (Driven) | Aplicação → Infraestrutura | Infraestrutura | `OrderRepository`, `EmailGateway`, `EventPublisher` |

**Regras obrigatórias**:
- Definido na camada **domínio** ou **aplicação** (nunca na infra)
- Usa apenas tipos do domínio (sem imports de SQLAlchemy, requests, etc.)
- Nome descreve a **intenção**, não a tecnologia
- Convenções: `*Repository`, `*Gateway`, `*Notifier`, `*Publisher`, `*Client`

---

## Template: Secondary Port — Repository

```python
# src/[context]/domain/repositories/[entity]_repository.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.[entity] import [Entity]
from ..value_objects.[id_vo] import [EntityId]


class [Entity]Repository(ABC):
    """
    Port secundário: contrato de persistência para [Entity].

    Quem implementa: infraestrutura (SQLAlchemy, MongoDB, in-memory, etc.)
    Quem usa: camada de aplicação (Use Cases)

    IMPORTANTE: Este arquivo pertence ao domínio.
    Não deve importar nada de infraestrutura.
    """

    @abstractmethod
    def find_by_id(self, entity_id: [EntityId]) -> Optional[[Entity]]:
        """Busca [Entity] pelo ID único. Retorna None se não encontrado."""
        ...

    @abstractmethod
    def find_all(self, page: int = 1, page_size: int = 20) -> List[[Entity]]:
        """Lista [entities] com paginação."""
        ...

    @abstractmethod
    def exists(self, entity_id: [EntityId]) -> bool:
        """Verifica existência sem carregar a entidade completa."""
        ...

    @abstractmethod
    def save(self, entity: [Entity]) -> None:
        """Persiste [Entity] (insert ou update)."""
        ...

    @abstractmethod
    def delete(self, entity_id: [EntityId]) -> None:
        """Remove [Entity]. Silencioso se não existir."""
        ...
```

---

## Template: Secondary Port — Gateway (Serviço Externo)

```python
# src/[context]/domain/ports/[service]_gateway.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class [ServiceRequest]:
    """Dados necessários para a chamada ao serviço externo."""
    [campo]: [tipo]


@dataclass(frozen=True)
class [ServiceResponse]:
    """Dados retornados pelo serviço externo."""
    [campo]: [tipo]
    success: bool


class [ServiceName]Gateway(ABC):
    """
    Port secundário: integração com [nome do serviço externo].

    Quem implementa: infraestrutura (HTTP client, SDK, etc.)
    Quem usa: camada de aplicação (Use Cases)

    Exemplos: EmailGateway, PaymentGateway, StorageGateway, SMSGateway
    """

    @abstractmethod
    def send(self, request: [ServiceRequest]) -> [ServiceResponse]:
        """[Descreva a operação com clareza]."""
        ...
```

### Exemplos concretos de Gateways

```python
# Port para envio de email
class EmailGateway(ABC):
    @abstractmethod
    def send_notification(self, to: str, subject: str, body: str) -> None: ...

    @abstractmethod
    def send_alert(self, to: list[str], alert: SecurityAlert) -> None: ...


# Port para armazenamento de arquivos
class FileStorageGateway(ABC):
    @abstractmethod
    def upload(self, file_path: str, destination: str) -> str: ...  # retorna URL

    @abstractmethod
    def delete(self, file_url: str) -> None: ...


# Port para detecção (domínio do projeto VisionGuard)
class DetectorPort(ABC):
    @abstractmethod
    def analyze(self, video_path: str) -> DetectionResult: ...

    @abstractmethod
    def clean_memory(self) -> None: ...
```

---

## Template: Secondary Port — Notifier

```python
# src/[context]/domain/ports/[entity]_notifier.py
from abc import ABC, abstractmethod
from ..events.[event] import [DomainEvent]


class [Context]Notifier(ABC):
    """
    Port secundário: notificação de eventos para sistemas externos.

    Implementações possíveis: webhook, email, Slack, SNS, SQS, etc.
    """

    @abstractmethod
    def notify(self, event: [DomainEvent]) -> None:
        """Envia notificação baseada no evento de domínio."""
        ...
```

---

## Template: Secondary Port — Event Publisher

```python
# src/[context]/domain/ports/event_publisher.py
from abc import ABC, abstractmethod
from ..events.base import DomainEvent


class EventPublisher(ABC):
    """
    Port secundário: publicação de Domain Events.

    Implementações: in-process (lista), Redis Pub/Sub, RabbitMQ, Kafka, etc.
    """

    @abstractmethod
    def publish(self, event: DomainEvent) -> None: ...

    @abstractmethod
    def publish_all(self, events: list[DomainEvent]) -> None: ...
```

---

## Template: Primary Port (Use Case Interface)

Usado quando a camada de apresentação precisa de uma abstração (testabilidade):

```python
# src/[context]/application/ports/[use_case]_port.py
from abc import ABC, abstractmethod
from .dtos.[dto] import [RequestDTO], [ResponseDTO]


class [UseCaseName]Port(ABC):
    """
    Port primário: contrato do use case para a camada de apresentação.

    Geralmente não é necessário (apresentação pode depender do use case diretamente).
    Use quando há múltiplas implementações do mesmo caso de uso.
    """

    @abstractmethod
    def execute(self, request: [RequestDTO]) -> [ResponseDTO]: ...
```

---

## Mapeamento Port → Adapter (na Infra)

```
Domínio define:                   Infra implementa:
─────────────────────────────────────────────────────────────
OrderRepository (ABC)     →       SQLAlchemyOrderRepository
EmailGateway (ABC)        →       SendGridEmailGateway
FileStorageGateway (ABC)  →       S3FileStorageGateway
EventPublisher (ABC)      →       InMemoryEventPublisher (tests)
                                  RedisEventPublisher (prod)
```

---

## Checklist

- [ ] Arquivo em `domain/repositories/` ou `domain/ports/`
- [ ] Herda de `ABC` com `@abstractmethod`
- [ ] Sem imports de infra (SQLAlchemy, requests, boto3, etc.)
- [ ] Usa apenas tipos do domínio nos parâmetros e retornos
- [ ] Nome descreve intenção, não tecnologia
- [ ] Docstring explica quem implementa e quem usa

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/domain/port-interface.md
#file:.github/copilot/skills/hexagonal-architecture.md
#file:src/[context]/domain/entities/[entity].py

Crie o Port [PortName] para [descreva a responsabilidade].
Operações necessárias: [liste os métodos]
```
