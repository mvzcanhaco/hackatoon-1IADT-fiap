# Template: Repository Implementation (Infra)

## Quando Usar

Repositórios na infraestrutura:
- Implementam as interfaces definidas no domínio
- Fazem o mapeamento entre domínio e modelo de persistência
- Encapsulam toda a lógica de acesso a dados
- Nunca contêm lógica de negócio

---

## Template: SQLAlchemy Repository

```python
# src/[context]/infrastructure/repositories/[entity]_sqlalchemy_repository.py
from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...domain.entities.[entity] import [Entity]
from ...domain.repositories.[repo_interface] import [Entity]Repository
from ...domain.value_objects.[id_vo] import [EntityId]
from .[orm_models] import [EntityModel]

logger = logging.getLogger(__name__)


class SQLAlchemy[Entity]Repository([Entity]Repository):
    """
    Adapter: Implementação de [Entity]Repository usando SQLAlchemy.

    Responsabilidade: Persistência e recuperação de [Entity].
    Tecnologia: SQLAlchemy ORM com PostgreSQL/SQLite.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ─── Queries ──────────────────────────────────────────────────────────────

    def find_by_id(self, entity_id: [EntityId]) -> Optional[[Entity]]:
        """Busca [Entity] pelo ID. Retorna None se não encontrado."""
        stmt = select([EntityModel]).where([EntityModel].id == str(entity_id))
        record = self._session.execute(stmt).scalar_one_or_none()

        if record is None:
            logger.debug("[Entity] not found: id=%s", entity_id)
            return None

        return self._to_domain(record)

    def find_by_[campo](self, [campo]: [tipo]) -> Optional[[Entity]]:
        """Busca [Entity] pelo [campo]. Retorna None se não encontrado."""
        stmt = select([EntityModel]).where([EntityModel].[campo] == [campo])
        record = self._session.execute(stmt).scalar_one_or_none()
        return self._to_domain(record) if record else None

    def find_all(
        self,
        page: int = 1,
        page_size: int = 20,
    ) -> List[[Entity]]:
        """Lista [entities] com paginação."""
        offset = (page - 1) * page_size
        stmt = (
            select([EntityModel])
            .order_by([EntityModel].created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        records = self._session.execute(stmt).scalars().all()
        return [self._to_domain(r) for r in records]

    def count(self) -> int:
        """Conta o total de [entities]."""
        from sqlalchemy import func
        stmt = select(func.count()).select_from([EntityModel])
        return self._session.execute(stmt).scalar_one()

    # ─── Comandos ─────────────────────────────────────────────────────────────

    def save(self, entity: [Entity]) -> None:
        """
        Persiste [Entity] (insert ou update).
        Usa merge para upsert automático.
        """
        record = self._to_persistence(entity)
        self._session.merge(record)
        logger.debug("[Entity] saved: id=%s", entity.id)

    def delete(self, entity_id: [EntityId]) -> None:
        """Remove [Entity] pelo ID. Silencioso se não existir."""
        stmt = select([EntityModel]).where([EntityModel].id == str(entity_id))
        record = self._session.execute(stmt).scalar_one_or_none()
        if record:
            self._session.delete(record)
            logger.debug("[Entity] deleted: id=%s", entity_id)

    # ─── Mapeamento Domínio ↔ Persistência ───────────────────────────────────

    def _to_domain(self, record: [EntityModel]) -> [Entity]:
        """
        Mapeia registro de banco para entidade de domínio.

        IMPORTANTE: Este método é a fronteira entre o mundo ORM e o domínio.
        O domínio nunca vê o modelo SQLAlchemy.
        """
        return [Entity](
            id=[EntityId](record.id),
            [campo]=record.[campo],
            # campos privados precisam de tratamento especial:
            # usar object.__setattr__ ou __init__ com todos os campos
        )

    def _to_persistence(self, entity: [Entity]) -> [EntityModel]:
        """
        Mapeia entidade de domínio para modelo SQLAlchemy.

        IMPORTANTE: Acessa propriedades públicas da entidade.
        Nunca acessa estado interno privado diretamente.
        """
        return [EntityModel](
            id=str(entity.id),
            [campo]=entity.[campo],
            status=entity.status.value,
            created_at=entity.created_at,
        )
```

---

## Template: ORM Model (SQLAlchemy)

```python
# src/[context]/infrastructure/repositories/[entity]_model.py
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from ..config.database import Base


class [EntityModel](Base):
    """
    Modelo ORM para [Entity].

    IMPORTANTE: Este é um modelo de persistência, não uma entidade de domínio.
    Não tem comportamento — apenas mapeamento de colunas.
    """

    __tablename__ = "[table_name]"

    id = Column(String(36), primary_key=True)
    [campo] = Column([SQLAlchemyType], nullable=False)
    status = Column(
        Enum(*[s.value for s in [EntityStatus]]),
        nullable=False,
        default=[EntityStatus].INITIAL.value,
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)

    # Relacionamentos
    [items] = relationship("[ItemModel]", back_populates="[entity]", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<[EntityModel] id={self.id} status={self.status}>"
```

---

## Template: In-Memory Repository (para Testes)

```python
# tests/fakes/in_memory_[entity]_repository.py
from typing import Dict, List, Optional

from src.[context].domain.entities.[entity] import [Entity]
from src.[context].domain.repositories.[repo_interface] import [Entity]Repository
from src.[context].domain.value_objects.[id_vo] import [EntityId]


class InMemory[Entity]Repository([Entity]Repository):
    """
    Implementação in-memory de [Entity]Repository para testes.

    Permite testes rápidos sem banco de dados real.
    Respeita o mesmo contrato da implementação de produção.
    """

    def __init__(self) -> None:
        self._store: Dict[str, [Entity]] = {}

    def find_by_id(self, entity_id: [EntityId]) -> Optional[[Entity]]:
        return self._store.get(str(entity_id))

    def find_by_[campo](self, [campo]: [tipo]) -> Optional[[Entity]]:
        return next(
            (e for e in self._store.values() if e.[campo] == [campo]),
            None,
        )

    def find_all(self, page: int = 1, page_size: int = 20) -> List[[Entity]]:
        items = list(self._store.values())
        start = (page - 1) * page_size
        return items[start:start + page_size]

    def count(self) -> int:
        return len(self._store)

    def save(self, entity: [Entity]) -> None:
        self._store[str(entity.id)] = entity

    def delete(self, entity_id: [EntityId]) -> None:
        self._store.pop(str(entity_id), None)

    # Helpers para testes
    def clear(self) -> None:
        self._store.clear()

    def seed(self, *entities: [Entity]) -> None:
        for entity in entities:
            self.save(entity)
```

---

## Checklist

- [ ] Implementa a interface do domínio
- [ ] Mapeamento `_to_domain()` e `_to_persistence()` separados
- [ ] Sem lógica de negócio
- [ ] Logging de operações importantes
- [ ] InMemory Fake para testes
- [ ] Testes de integração com banco real (SQLite in-memory)

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/infrastructure/repository.md
#file:src/[context]/domain/repositories/[repo_interface].py
#file:src/[context]/domain/entities/[entity].py

Implemente [Entity]SQLAlchemyRepository com:
- Queries necessárias: [lista]
- Campos mapeados: [lista]
```
