# Template: Use Case (Caso de Uso)

## Quando Usar

Use cases ficam na camada de aplicação e:
- Orquestram o domínio para realizar uma operação de negócio
- Têm **uma única responsabilidade** e um único método `execute()`
- Recebem um **DTO de Request** e retornam um **DTO de Response**
- Dependem de **interfaces** (ports), nunca de implementações concretas
- Não contêm lógica de negócio — delegam ao domínio

---

## Template Completo

```python
# src/[context]/application/use_cases/[use_case_name].py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ...domain.entities.[entity] import [Entity]
from ...domain.interfaces.[port] import [PortInterface]
from ...domain.exceptions.[context]_exceptions import (
    [EntityNotFoundError],
    [BusinessRuleError],
)

logger = logging.getLogger(__name__)


# ─── DTOs ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class [UseCaseName]Request:
    """
    DTO de entrada para [UseCaseName].

    Contém apenas dados primitivos ou Value Objects simples.
    Sem lógica de negócio.
    """
    [campo_obrigatorio]: [tipo]           # [descrição]
    [campo_opcional]: Optional[[tipo]] = None  # [descrição], padrão: [valor]


@dataclass(frozen=True)
class [UseCaseName]Response:
    """
    DTO de saída para [UseCaseName].

    Retorna apenas o necessário para o caller.
    Nunca retorna entidades de domínio diretamente.
    """
    [campo_resultado]: [tipo]             # [descrição]
    [campo_opcional]: Optional[[tipo]] = None


# ─── Use Case ─────────────────────────────────────────────────────────────────

@dataclass
class [UseCaseName]:
    """
    Use Case: [UseCaseName]

    Responsabilidade: [descreva em uma frase o que este use case faz]

    Fluxo:
    1. [passo 1 — ex: busca a entidade pelo ID]
    2. [passo 2 — ex: executa a ação de domínio]
    3. [passo 3 — ex: persiste a mudança]
    4. [passo 4 — ex: notifica via serviço externo se necessário]
    5. [passo 5 — ex: retorna o response DTO]

    Dependências:
    - [InterfaceName]: [para que serve]
    """

    [dependency_name]: [InterfaceType]
    [another_dependency]: [AnotherInterfaceType]

    def execute(self, request: [UseCaseName]Request) -> [UseCaseName]Response:
        """
        Executa [nome da operação de negócio].

        Args:
            request: [descrição do que vem no request]

        Returns:
            [UseCaseName]Response: [descrição do que é retornado]

        Raises:
            [EntityNotFoundError]: quando [recurso] não existe
            [BusinessRuleError]: quando [regra de negócio é violada]
        """
        logger.info(
            "[UseCaseName] started",
            extra={"[campo_id]": request.[campo_obrigatorio]},
        )

        # 1. Buscar entidade principal
        [entity] = self.[dependency_name].find_by_[campo](request.[campo_obrigatorio])
        if [entity] is None:
            raise [EntityNotFoundError](request.[campo_obrigatorio])

        # 2. Executar ação de domínio (regra de negócio na entidade)
        event = [entity].[acao_de_negocio]([params_do_request])

        # 3. Persistir mudança
        self.[dependency_name].save([entity])

        # 4. Processar evento (notificar, publicar, etc.) — se necessário
        if event and self.[another_dependency]:
            self.[another_dependency].[process_event](event)

        logger.info(
            "[UseCaseName] completed",
            extra={"[campo_id]": str([entity].id)},
        )

        # 5. Retornar response DTO
        return [UseCaseName]Response(
            [campo_resultado]=[entity].[campo_resultado],
        )
```

---

## Template: Use Case com Criação (Create)

```python
@dataclass
class Create[Entity]UseCase:
    """Cria uma nova instância de [Entity]."""

    [entity]_repository: [Entity]Repository
    [optional_service]: Optional[[ServicePort]] = None

    def execute(self, request: Create[Entity]Request) -> Create[Entity]Response:
        logger.info("Creating [Entity]", extra={"[campo]": request.[campo]})

        # Verificar unicidade (se necessário)
        existing = self.[entity]_repository.find_by_[campo](request.[campo])
        if existing:
            raise [Entity]AlreadyExistsError(request.[campo])

        # Criar entidade via factory method
        [entity], created_event = [Entity].create(
            [campo]=request.[campo],
        )

        # Persistir
        self.[entity]_repository.save([entity])

        logger.info("[Entity] created", extra={"id": str([entity].id)})

        return Create[Entity]Response(
            [entity]_id=str([entity].id),
        )
```

---

## Template: Use Case de Lista/Query (Read)

```python
@dataclass
class List[Entities]UseCase:
    """Lista [entities] com filtros opcionais."""

    [entity]_repository: [Entity]Repository

    def execute(self, request: List[Entities]Request) -> List[Entities]Response:
        [entities] = self.[entity]_repository.find_all(
            filters=request.filters,
            page=request.page,
            page_size=request.page_size,
        )

        return List[Entities]Response(
            items=[self._to_dto(e) for e in [entities]],
            total=self.[entity]_repository.count(filters=request.filters),
            page=request.page,
            page_size=request.page_size,
        )

    def _to_dto(self, entity: [Entity]) -> [Entity]DTO:
        return [Entity]DTO(
            id=str(entity.id),
            [campo]=entity.[campo],
        )
```

---

## Checklist

- [ ] Um único método `execute(request) -> response`
- [ ] Request e Response são DTOs `frozen=True`
- [ ] Sem lógica de negócio (delegada ao domínio)
- [ ] Dependências como interfaces (ports) no construtor
- [ ] Logging de início e fim (com IDs relevantes)
- [ ] Exceções de domínio propagadas sem modificação
- [ ] Exceções de infraestrutura podem ser encapsuladas
- [ ] Testes unitários com todas as dependências mockadas

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/application/use-case.md
#file:src/domain/entities/[entity].py
#file:src/domain/interfaces/[port].py

Implemente o use case [NomeDoUseCase] que:
- Recebe: [descrição do input]
- Faz: [descrição do fluxo]
- Retorna: [descrição do output]
- Regras: [lista das regras de negócio]
```
