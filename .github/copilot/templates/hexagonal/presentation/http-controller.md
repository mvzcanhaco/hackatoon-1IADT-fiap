# Template: HTTP Controller (Presentation)

## Quando Usar

Controllers ficam na camada de apresentação e:
- São o ponto de entrada de requisições HTTP
- Validam o formato do input (schema validation)
- Chamam use cases via injeção de dependência
- Convertem erros de domínio em respostas HTTP adequadas
- Nunca contêm lógica de negócio

---

## Template: FastAPI Router

```python
# src/[context]/presentation/http/[resource]_router.py
from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...application.use_cases.create_[entity] import (
    Create[Entity]UseCase,
    Create[Entity]Request,
)
from ...application.use_cases.get_[entity] import (
    Get[Entity]UseCase,
    Get[Entity]Request,
)
from ...application.use_cases.list_[entities] import (
    List[Entities]UseCase,
    List[Entities]Request,
)
from ...domain.exceptions.[context]_exceptions import (
    [Entity]NotFoundError,
    [BusinessRuleError],
    [Entity]AlreadyExistsError,
)
from .dependencies import (
    get_create_[entity]_use_case,
    get_get_[entity]_use_case,
    get_list_[entities]_use_case,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/[resources]", tags=["[Resource]"])


# ─── Schemas Pydantic ────────────────────────────────────────────────────────

class Create[Entity]RequestSchema(BaseModel):
    """Payload para criação de [Entity]."""
    [campo_obrigatorio]: [tipo] = Field(
        ...,
        description="[descrição do campo]",
        example="[exemplo]",
    )
    [campo_opcional]: Optional[[tipo]] = Field(
        default=None,
        description="[descrição — opcional]",
    )


class [Entity]ResponseSchema(BaseModel):
    """Representação de [Entity] nas respostas."""
    id: str = Field(description="ID único da [entity]")
    [campo]: [tipo] = Field(description="[descrição]")
    status: str = Field(description="Status atual")

    class Config:
        from_attributes = True  # Pydantic v2 (ou orm_mode = True no v1)


class PaginatedResponse(BaseModel):
    """Response paginado genérico."""
    items: List[[Entity]ResponseSchema]
    total: int
    page: int
    page_size: int


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=[Entity]ResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Cria uma nova [entity]",
)
async def create_[entity](
    payload: Create[Entity]RequestSchema,
    use_case: Create[Entity]UseCase = Depends(get_create_[entity]_use_case),
) -> [Entity]ResponseSchema:
    """
    Cria uma nova [entity] com os dados fornecidos.

    - **[campo]**: [descrição do campo]

    **Erros possíveis:**
    - `422`: dados inválidos ou regra de negócio violada
    - `409`: [entity] já existe com este [campo único]
    """
    try:
        response = use_case.execute(
            Create[Entity]Request(
                [campo_obrigatorio]=payload.[campo_obrigatorio],
                [campo_opcional]=payload.[campo_opcional],
            )
        )
        return [Entity]ResponseSchema(
            id=response.[entity]_id,
            [campo]=payload.[campo_obrigatorio],
            status="active",
        )
    except [Entity]AlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except [BusinessRuleError] as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.get(
    "/{[entity]_id}",
    response_model=[Entity]ResponseSchema,
    summary="Busca uma [entity] pelo ID",
)
async def get_[entity](
    [entity]_id: UUID,
    use_case: Get[Entity]UseCase = Depends(get_get_[entity]_use_case),
) -> [Entity]ResponseSchema:
    """Retorna os dados de uma [entity] específica."""
    try:
        response = use_case.execute(
            Get[Entity]Request([entity]_id=str([entity]_id))
        )
        return [Entity]ResponseSchema(
            id=response.id,
            [campo]=response.[campo],
            status=response.status,
        )
    except [Entity]NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"[Entity] not found: {[entity]_id}",
        )


@router.get(
    "/",
    response_model=PaginatedResponse,
    summary="Lista [entities]",
)
async def list_[entities](
    page: int = Query(default=1, ge=1, description="Número da página"),
    page_size: int = Query(default=20, ge=1, le=100, description="Itens por página"),
    use_case: List[Entities]UseCase = Depends(get_list_[entities]_use_case),
) -> PaginatedResponse:
    """Lista [entities] com paginação."""
    response = use_case.execute(
        List[Entities]Request(page=page, page_size=page_size)
    )
    return PaginatedResponse(
        items=[
            [Entity]ResponseSchema(id=item.id, [campo]=item.[campo], status=item.status)
            for item in response.items
        ],
        total=response.total,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/{[entity]_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove uma [entity]",
)
async def delete_[entity](
    [entity]_id: UUID,
    use_case: Delete[Entity]UseCase = Depends(get_delete_[entity]_use_case),
) -> None:
    """Remove uma [entity] permanentemente."""
    try:
        use_case.execute(Delete[Entity]Request([entity]_id=str([entity]_id)))
    except [Entity]NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"[Entity] not found: {[entity]_id}",
        )
```

---

## Template: Dependencies (DI)

```python
# src/[context]/presentation/http/dependencies.py
from functools import lru_cache
from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from ...application.use_cases.create_[entity] import Create[Entity]UseCase
from ...infrastructure.config.database import get_db_session
from ...infrastructure.repositories.[entity]_sqlalchemy_repository import (
    SQLAlchemy[Entity]Repository,
)


def get_create_[entity]_use_case(
    session: Session = Depends(get_db_session),
) -> Create[Entity]UseCase:
    """
    Factory function para injeção do use case.
    Monta o grafo completo de dependências.
    """
    repository = SQLAlchemy[Entity]Repository(session)
    return Create[Entity]UseCase([entity]_repository=repository)
```

---

## Mapeamento de Erros de Domínio → HTTP

| Exceção de Domínio | HTTP Status | Quando |
|---------------------|------------|--------|
| `[Entity]NotFoundError` | 404 Not Found | Recurso não existe |
| `[Entity]AlreadyExistsError` | 409 Conflict | Duplicata |
| `[BusinessRuleError]` | 422 Unprocessable Entity | Regra de negócio |
| `[InvalidStateTransitionError]` | 422 Unprocessable Entity | Estado inválido |
| `UnauthorizedError` | 403 Forbidden | Sem permissão |
| `ValidationError` (Pydantic) | 400 Bad Request | Dados malformados |

---

## Checklist

- [ ] Schema Pydantic para validação de input
- [ ] Schema de response sem dados sensíveis
- [ ] Todos os erros de domínio mapeados para HTTP adequado
- [ ] Nenhum acesso direto a repositório ou banco
- [ ] Logging de erros inesperados
- [ ] Documentação Swagger/OpenAPI nos endpoints
- [ ] Paginação em endpoints de lista

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/presentation/http-controller.md
#file:src/[context]/application/use_cases/[use_case].py

Crie o router HTTP para [recurso] com endpoints:
- POST /[resources] — criar
- GET /[resources]/{id} — buscar
- GET /[resources] — listar
- DELETE /[resources]/{id} — remover
```
