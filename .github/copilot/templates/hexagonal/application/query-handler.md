# Template: Query Handler (CQRS Read Side)

> Referência: [CQRS Pattern — Martin Fowler](https://martinfowler.com/bliki/CQRS.html), [Greg Young — CQRS Documents](https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf)

---

## Quando Usar Query Handler vs Use Case

```
Use Case (Command):
  ✅ Altera estado (criar, confirmar, cancelar)
  ✅ Publica domain events
  ✅ Passa pelo aggregate root para garantir invariantes
  ✅ Uma transação de banco

Query Handler (Query):
  ✅ Leitura pura — zero efeito colateral
  ✅ Pode agregar dados de múltiplas fontes (ex: join de tabelas)
  ✅ Pode ter model de leitura otimizado (read model / projection)
  ✅ Não passa pelo aggregate root (não precisa das invariantes)
  ✅ Pode usar cache agressivamente
  ❌ NUNCA altera estado
  ❌ NUNCA publica eventos
```

---

## Template: Query + Response DTO

```python
# src/application/queries/[query_name].py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class [QueryName]Query:
    """
    Parâmetros de leitura para [recurso].
    Todos os campos são opcionais (filtros) ou identificadores.
    """
    [filtro_1]: [Tipo]
    [filtro_2]: Optional[[Tipo]] = None
    page: int = 1
    page_size: int = 20


@dataclass(frozen=True)
class [ItemDTO]:
    """DTO de leitura — não é a entidade de domínio."""
    id: str
    [campo_1]: [Tipo]
    [campo_2]: [Tipo]
    created_at: datetime


@dataclass(frozen=True)
class [QueryName]Response:
    """Resposta paginada da query."""
    items: list[[ItemDTO]]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return max(1, -(-self.total // self.page_size))  # ceil division

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages
```

---

## Template: Query Handler

```python
# src/application/queries/[query_name]_handler.py
import logging
from src.application.queries.[query_name] import (
    [QueryName]Query,
    [QueryName]Response,
    [ItemDTO],
)
from src.domain.ports.[resource]_read_port import [Resource]ReadPort

logger = logging.getLogger(__name__)


class [QueryName]Handler:
    """
    Handler de leitura para [recurso].
    Não usa o aggregate — acessa o read model diretamente.
    """

    def __init__(self, read_port: [Resource]ReadPort) -> None:
        self._read_port = read_port

    def handle(self, query: [QueryName]Query) -> [QueryName]Response:
        logger.info(
            "[QueryName] executed",
            extra={"[filtro_1]": query.[filtro_1], "page": query.page},
        )

        items, total = self._read_port.find_by_[filtro_1](
            [filtro_1]=query.[filtro_1],
            [filtro_2]=query.[filtro_2],
            offset=(query.page - 1) * query.page_size,
            limit=query.page_size,
        )

        return [QueryName]Response(
            items=[self._to_dto(item) for item in items],
            total=total,
            page=query.page,
            page_size=query.page_size,
        )

    @staticmethod
    def _to_dto(raw: dict) -> [ItemDTO]:
        return [ItemDTO](
            id=str(raw["id"]),
            [campo_1]=raw["[campo_1]"],
            [campo_2]=raw["[campo_2]"],
            created_at=raw["created_at"],
        )
```

---

## Template: Read Port (Secondary Port para leitura)

```python
# src/domain/ports/[resource]_read_port.py
from abc import ABC, abstractmethod
from typing import Optional


class [Resource]ReadPort(ABC):
    """
    Secondary Port de leitura para [Resource].
    Separado do Repository (que usa aggregate root).
    Pode ser otimizado com SQL nativo, views materializadas, etc.
    """

    @abstractmethod
    def find_by_[filtro_1](
        self,
        [filtro_1]: [Tipo],
        [filtro_2]: Optional[[Tipo]],
        offset: int,
        limit: int,
    ) -> tuple[list[dict], int]:
        """
        Retorna (items_raw, total_count).
        items_raw: lista de dicts com os campos do DTO.
        """
        ...

    @abstractmethod
    def find_by_id(self, id: str) -> Optional[dict]:
        """Retorna dict com campos do DTO ou None."""
        ...
```

---

## Template: Read Adapter (SQL nativo)

```python
# src/infrastructure/read_models/[resource]_sql_read_adapter.py
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.domain.ports.[resource]_read_port import [Resource]ReadPort


class [Resource]SqlReadAdapter([Resource]ReadPort):
    """
    Implementação SQL nativa do read model.
    Pode usar JOINs, views materializadas, subqueries — sem passar pelo ORM do aggregate.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_[filtro_1](
        self,
        [filtro_1]: [Tipo],
        [filtro_2]: Optional[[Tipo]],
        offset: int,
        limit: int,
    ) -> tuple[list[dict], int]:
        filters = "WHERE r.[filtro_1] = :[filtro_1]"
        params: dict = {"[filtro_1]": str([filtro_1]), "offset": offset, "limit": limit}

        if [filtro_2] is not None:
            filters += " AND r.[filtro_2] = :[filtro_2]"
            params["[filtro_2]"] = str([filtro_2])

        rows = self._session.execute(
            text(f"""
                SELECT
                    r.id,
                    r.[campo_1],
                    r.[campo_2],
                    r.created_at
                FROM [tabela] r
                {filters}
                ORDER BY r.created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).mappings().all()

        total = self._session.execute(
            text(f"SELECT COUNT(*) FROM [tabela] r {filters}"),
            {k: v for k, v in params.items() if k not in ("offset", "limit")},
        ).scalar_one()

        return [dict(row) for row in rows], total

    def find_by_id(self, id: str) -> Optional[dict]:
        row = self._session.execute(
            text("""
                SELECT id, [campo_1], [campo_2], created_at
                FROM [tabela]
                WHERE id = :id
            """),
            {"id": id},
        ).mappings().one_or_none()

        return dict(row) if row else None
```

---

## Exemplo Concreto: ListOrdersQuery

```python
# src/application/queries/list_orders.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class ListOrdersQuery:
    customer_id: str
    status: Optional[str] = None
    page: int = 1
    page_size: int = 20


@dataclass(frozen=True)
class OrderSummaryDTO:
    id: str
    status: str
    total_cents: int
    item_count: int
    created_at: datetime


@dataclass(frozen=True)
class ListOrdersResponse:
    items: list[OrderSummaryDTO]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        return max(1, -(-self.total // self.page_size))


# src/application/queries/list_orders_handler.py
class ListOrdersHandler:
    def __init__(self, order_read_port: OrderReadPort) -> None:
        self._port = order_read_port

    def handle(self, query: ListOrdersQuery) -> ListOrdersResponse:
        rows, total = self._port.find_by_customer(
            customer_id=query.customer_id,
            status=query.status,
            offset=(query.page - 1) * query.page_size,
            limit=query.page_size,
        )
        return ListOrdersResponse(
            items=[
                OrderSummaryDTO(
                    id=str(r["id"]),
                    status=r["status"],
                    total_cents=r["total_cents"],
                    item_count=r["item_count"],
                    created_at=r["created_at"],
                )
                for r in rows
            ],
            total=total,
            page=query.page,
            page_size=query.page_size,
        )
```

---

## Cache em Query Handlers

```python
import json
from typing import Optional

class CachedListOrdersHandler:
    """Decorator de cache sobre o handler real."""

    TTL = 60  # 1 minuto

    def __init__(self, handler: ListOrdersHandler, redis) -> None:
        self._handler = handler
        self._redis = redis

    def handle(self, query: ListOrdersQuery) -> ListOrdersResponse:
        cache_key = f"orders:list:{query.customer_id}:{query.status}:p{query.page}:s{query.page_size}"

        cached = self._redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return ListOrdersResponse(**data)

        result = self._handler.handle(query)

        self._redis.setex(cache_key, self.TTL, json.dumps(result.__dict__, default=str))
        return result
```

---

## Checklist do Query Handler

```
Design:
  [ ] Query é imutável (frozen=True) — parâmetros de busca
  [ ] Response é imutável (frozen=True) — dados de retorno
  [ ] Handler não usa repositório do aggregate — usa Read Port separado
  [ ] Read adapter pode usar SQL nativo, JOIN, view materializada

Separação de responsabilidades:
  [ ] Query Handler NUNCA altera estado
  [ ] Query Handler NUNCA publica eventos
  [ ] Paginação sempre incluída em listagens (page + page_size + total)

Performance:
  [ ] Read model pode ser desnormalizado (soma de itens pré-calculada, etc.)
  [ ] Cache opcional no handler (com TTL proporcional à frequência de mudança)
  [ ] SQL com índices verificados para os filtros mais comuns
```

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/application/query-handler.md
#file:.github/copilot/templates/hexagonal/application/use-case.md
#file:src/domain/entities/[entidade].py

Implemente a query [QueryName] para listar [recurso] com filtros:
  - [filtro 1]: [tipo e comportamento]
  - [filtro 2]: [opcional/obrigatório]
Paginação: [offset | cursor]
Cache: [sim/não, TTL sugerido]
```
