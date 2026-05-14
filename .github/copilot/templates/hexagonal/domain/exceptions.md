# Template: Exception Hierarchy (Domínio)

## Quando Usar

Exceções de domínio comunicam **violações de regras de negócio** de forma expressiva, substituindo mensagens de erro genéricas.

**Regras obrigatórias**:
- Definidas na camada de **domínio** (sem imports de framework)
- Nome descreve o **problema de negócio**, não o erro técnico
- Carregam contexto suficiente para logging e debugging
- Mapeadas para HTTP status codes na camada de **apresentação**
- Nunca use `Exception` ou `ValueError` para erros de domínio

---

## Hierarquia Completa

```python
# src/[context]/domain/exceptions/__init__.py

DomainException                        ← base abstrata
├── EntityNotFoundError                ← 404 Not Found
├── BusinessRuleViolationError         ← 422 Unprocessable Entity
│   ├── InvalidStateTransitionError    ← operação inválida para o estado atual
│   ├── InvariantViolationError        ← invariante do agregado violada
│   └── DuplicateEntityError           ← tentativa de criar duplicata
├── InvalidValueError                  ← 400 Bad Request (dados inválidos)
│   ├── InvalidFormatError             ← formato inválido (email, UUID, etc.)
│   └── ValueOutOfRangeError           ← valor fora dos limites permitidos
└── AggregateConflictError             ← 409 Conflict (concorrência)
```

---

## Template: Base Exception

```python
# src/[context]/domain/exceptions/domain_exception.py
from __future__ import annotations


class DomainException(Exception):
    """
    Classe base para todas as exceções de domínio.

    Exceções de domínio representam violações de regras de negócio,
    não erros técnicos. Nunca devem conter imports de framework.
    """

    def __init__(
        self,
        message: str,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{super().__str__()} [{ctx}]"
        return super().__str__()
```

---

## Template: EntityNotFoundError

```python
# src/[context]/domain/exceptions/not_found.py
from .domain_exception import DomainException


class EntityNotFoundError(DomainException):
    """
    Entidade não encontrada pelo identificador fornecido.

    Mapeamento HTTP: 404 Not Found
    """

    def __init__(self, entity_name: str, entity_id: object) -> None:
        super().__init__(
            f"{entity_name} not found",
            context={"entity": entity_name, "id": str(entity_id)},
        )
        self.entity_name = entity_name
        self.entity_id = entity_id


# Exemplos de uso específicos — crie um por tipo de entidade:

class OrderNotFoundError(EntityNotFoundError):
    def __init__(self, order_id: object) -> None:
        super().__init__("Order", order_id)


class VideoNotFoundError(EntityNotFoundError):
    def __init__(self, video_id: object) -> None:
        super().__init__("Video", video_id)
```

---

## Template: BusinessRuleViolationError

```python
# src/[context]/domain/exceptions/business_rule.py
from .domain_exception import DomainException


class BusinessRuleViolationError(DomainException):
    """
    Regra de negócio violada.

    Mapeamento HTTP: 422 Unprocessable Entity
    """

    def __init__(self, rule: str, context: dict[str, object] | None = None) -> None:
        super().__init__(f"Business rule violated: {rule}", context)
        self.rule = rule


class InvalidStateTransitionError(BusinessRuleViolationError):
    """
    Operação inválida para o estado atual da entidade.

    Exemplo: tentar confirmar um pedido já cancelado.
    Mapeamento HTTP: 422 Unprocessable Entity
    """

    def __init__(
        self,
        entity: str,
        current_state: object,
        attempted_action: str,
    ) -> None:
        super().__init__(
            f"Cannot {attempted_action} a {entity} in state {current_state!r}",
            context={
                "entity": entity,
                "current_state": str(current_state),
                "attempted_action": attempted_action,
            },
        )


class DuplicateEntityError(BusinessRuleViolationError):
    """
    Tentativa de criar entidade que já existe.

    Mapeamento HTTP: 409 Conflict
    """

    def __init__(self, entity_name: str, identifier: object) -> None:
        super().__init__(
            f"{entity_name} already exists",
            context={"entity": entity_name, "identifier": str(identifier)},
        )


class InvariantViolationError(BusinessRuleViolationError):
    """
    Invariante do agregado violada.

    Exemplo: tentar confirmar pedido sem itens.
    Mapeamento HTTP: 422 Unprocessable Entity
    """

    def __init__(self, invariant: str, context: dict[str, object] | None = None) -> None:
        super().__init__(f"Invariant violated: {invariant}", context)
```

---

## Template: InvalidValueError

```python
# src/[context]/domain/exceptions/invalid_value.py
from .domain_exception import DomainException


class InvalidValueError(DomainException):
    """
    Valor inválido fornecido para um Value Object ou campo.

    Mapeamento HTTP: 400 Bad Request
    """

    def __init__(
        self,
        field_name: str,
        value: object,
        reason: str,
    ) -> None:
        super().__init__(
            f"Invalid value for {field_name}: {reason}",
            context={"field": field_name, "value": str(value), "reason": reason},
        )


class InvalidFormatError(InvalidValueError):
    """Formato inválido (email, UUID, URL, etc.)."""

    def __init__(self, field_name: str, value: object, expected_format: str) -> None:
        super().__init__(
            field_name,
            value,
            reason=f"expected format: {expected_format}",
        )


class ValueOutOfRangeError(InvalidValueError):
    """Valor fora dos limites permitidos."""

    def __init__(
        self,
        field_name: str,
        value: object,
        min_value: object | None = None,
        max_value: object | None = None,
    ) -> None:
        bounds = []
        if min_value is not None:
            bounds.append(f"min={min_value}")
        if max_value is not None:
            bounds.append(f"max={max_value}")
        super().__init__(
            field_name,
            value,
            reason=f"out of range [{', '.join(bounds)}]",
        )
```

---

## Template: AggregateConflictError

```python
# src/[context]/domain/exceptions/conflict.py
from .domain_exception import DomainException


class AggregateConflictError(DomainException):
    """
    Conflito de concorrência — aggregate modificado por outra operação.

    Mapeamento HTTP: 409 Conflict
    Estratégia: retry com backoff ou informar usuário para recarregar
    """

    def __init__(self, entity_name: str, entity_id: object) -> None:
        super().__init__(
            f"Conflict: {entity_name} was modified concurrently",
            context={"entity": entity_name, "id": str(entity_id)},
        )
```

---

## Arquivo de Init — Exports Públicos

```python
# src/[context]/domain/exceptions/__init__.py
from .domain_exception import DomainException
from .not_found import EntityNotFoundError, OrderNotFoundError
from .business_rule import (
    BusinessRuleViolationError,
    InvalidStateTransitionError,
    DuplicateEntityError,
    InvariantViolationError,
)
from .invalid_value import InvalidValueError, InvalidFormatError, ValueOutOfRangeError
from .conflict import AggregateConflictError

__all__ = [
    "DomainException",
    "EntityNotFoundError",
    "OrderNotFoundError",
    "BusinessRuleViolationError",
    "InvalidStateTransitionError",
    "DuplicateEntityError",
    "InvariantViolationError",
    "InvalidValueError",
    "InvalidFormatError",
    "ValueOutOfRangeError",
    "AggregateConflictError",
]
```

---

## Mapeamento Domain → HTTP (na Apresentação)

```python
# src/[context]/presentation/http/error_mapper.py
from fastapi import HTTPException, status
from ...domain.exceptions import (
    EntityNotFoundError,
    BusinessRuleViolationError,
    DuplicateEntityError,
    InvalidValueError,
    AggregateConflictError,
    DomainException,
)


def to_http_exception(exc: DomainException) -> HTTPException:
    """Mapeia exceções de domínio para respostas HTTP semânticas."""
    if isinstance(exc, EntityNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc))

    if isinstance(exc, DuplicateEntityError | AggregateConflictError):
        return HTTPException(status.HTTP_409_CONFLICT, detail=str(exc))

    if isinstance(exc, InvalidValueError):
        return HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if isinstance(exc, BusinessRuleViolationError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    # Fallback: DomainException genérica → 422
    return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
```

### Uso no Controller

```python
@router.post("/orders")
async def create_order(
    payload: CreateOrderSchema,
    use_case: CreateOrderUseCase = Depends(get_create_order),
) -> OrderResponse:
    try:
        result = use_case.execute(CreateOrderRequest(...))
        return OrderResponse(...)
    except DomainException as exc:
        raise to_http_exception(exc) from exc
```

---

## Checklist

- [ ] Hierarquia herda de `DomainException` (nunca de `Exception` diretamente)
- [ ] Sem imports de frameworks externos
- [ ] Contexto rico: `entity`, `id`, `rule`, `field` conforme o caso
- [ ] Nome descreve o problema de negócio (não o erro técnico)
- [ ] `__init__.py` com exports explícitos
- [ ] Mapeador HTTP na camada de apresentação
- [ ] Testes de domínio verificam o tipo exato da exceção (`pytest.raises`)

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/domain/exceptions.md
#file:src/[context]/domain/entities/[entity].py

Crie a hierarquia de exceções para o contexto [ContextName].
Exceções necessárias:
- [EntityName]NotFoundError
- [Rule]ViolationError (quando [condição])
- [Field]InvalidError (quando [condição])
```
