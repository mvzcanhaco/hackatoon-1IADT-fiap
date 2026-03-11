# Template: Value Object

## Quando Usar

Use quando o objeto:
- É descrito pelos seus atributos (não por identidade)
- Dois objetos com os mesmos atributos são iguais
- Deve ser imutável (não muda após criação)
- Encapsula validação e lógica de um conceito do domínio

**Candidatos típicos**: Email, CPF, CNPJ, Money, Phone, Address, DateRange,
Percentage, Coordinate, Color, Temperature, URL, OrderId, UserId...

---

## Template: VO Simples (um atributo)

```python
# src/[context]/domain/value_objects/[name].py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class [ValueObjectName]:
    """
    Value Object: [ValueObjectName]

    Representa [descreva o conceito de domínio].
    Imutável: sim (frozen=True)
    Igualdade: por valor (não por referência)

    Exemplos válidos: [ex1], [ex2]
    Exemplos inválidos: [ex3] (razão), [ex4] (razão)
    """

    value: [tipo_primitivo]  # str, int, float, Decimal

    # Constante de validação (se aplicável)
    _PATTERN: ClassVar[re.Pattern] = re.compile(r"[regex_de_validação]")

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.value:
            raise ValueError(f"[ValueObjectName] cannot be empty")
        if not self._PATTERN.match(str(self.value)):
            raise ValueError(
                f"Invalid [ValueObjectName]: '{self.value}'. "
                f"Expected format: [descreva o formato esperado]"
            )

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"[ValueObjectName]('{self.value}')"
```

---

## Template: VO Composto (múltiplos atributos)

```python
# src/[context]/domain/value_objects/money.py
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import ClassVar, FrozenSet


@dataclass(frozen=True)
class Money:
    """
    Value Object: Money

    Representa um valor monetário com moeda explícita.
    Imutável: sim. Operações retornam novos objetos.

    Exemplos:
        price = Money(Decimal("29.99"), "BRL")
        total = price.add(Money(Decimal("5.00"), "BRL"))
        discounted = price.multiply(Decimal("0.9"))  # 10% desconto
    """

    amount: Decimal
    currency: str

    VALID_CURRENCIES: ClassVar[FrozenSet[str]] = frozenset(["BRL", "USD", "EUR"])
    ZERO_BRL: ClassVar["Money"]  # definido após a classe

    def __post_init__(self) -> None:
        # frozen=True impede setattr, mas podemos validar aqui
        object.__setattr__(self, "amount", self.amount.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ))
        self._validate()

    def _validate(self) -> None:
        if self.amount < Decimal("0"):
            raise ValueError(f"Money amount cannot be negative: {self.amount}")
        if self.currency not in self.VALID_CURRENCIES:
            raise ValueError(
                f"Invalid currency: '{self.currency}'. "
                f"Valid: {', '.join(sorted(self.VALID_CURRENCIES))}"
            )

    # ─── Operações Aritméticas ────────────────────────────────────────────────

    def add(self, other: "Money") -> "Money":
        """Soma dois valores monetários da mesma moeda."""
        self._assert_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def subtract(self, other: "Money") -> "Money":
        """Subtrai, mantendo invariante de não-negatividade."""
        self._assert_same_currency(other)
        result = self.amount - other.amount
        if result < Decimal("0"):
            raise ValueError(f"Money subtraction would result in negative amount")
        return Money(result, self.currency)

    def multiply(self, factor: Decimal) -> "Money":
        """Multiplica por um fator (ex: desconto, quantidade)."""
        return Money(self.amount * factor, self.currency)

    # ─── Comparações ─────────────────────────────────────────────────────────

    def is_greater_than(self, other: "Money") -> bool:
        self._assert_same_currency(other)
        return self.amount > other.amount

    def is_zero(self) -> bool:
        return self.amount == Decimal("0")

    # ─── Formatação ──────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"

    def __repr__(self) -> str:
        return f"Money(amount={self.amount!r}, currency={self.currency!r})"

    # ─── Helpers Privados ─────────────────────────────────────────────────────

    def _assert_same_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot operate on different currencies: "
                f"{self.currency} and {other.currency}"
            )


Money.ZERO_BRL = Money(Decimal("0"), "BRL")
```

---

## Template: VO de Status / Estado

```python
# src/[context]/domain/value_objects/[entity]_status.py
from enum import Enum, auto
from typing import FrozenSet


class [Entity]Status(Enum):
    """
    Value Object: [Entity]Status

    Representa os estados possíveis de [Entity].
    Transições válidas documentadas em ALLOWED_TRANSITIONS.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

    # ─── Transições Permitidas ────────────────────────────────────────────────

    ALLOWED_TRANSITIONS: dict = {
        # chave: estado atual → valor: estados para os quais pode transicionar
    }

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"Invalid [Entity]Status: '{value}'")

    def can_transition_to(self, target: "[Entity]Status") -> bool:
        """Verifica se a transição é permitida."""
        allowed = self.ALLOWED_TRANSITIONS.get(self, frozenset())
        return target in allowed

    def transition_to(self, target: "[Entity]Status") -> "[Entity]Status":
        """Executa a transição ou lança exceção se inválida."""
        if not self.can_transition_to(target):
            raise InvalidStatusTransitionError(
                current=self,
                target=target,
            )
        return target

    @classmethod
    def editable_states(cls) -> FrozenSet["[Entity]Status"]:
        """Estados em que a entidade pode ser modificada."""
        return frozenset({cls.DRAFT})

    @classmethod
    def terminal_states(cls) -> FrozenSet["[Entity]Status"]:
        """Estados finais — entidade não pode mais mudar."""
        return frozenset({cls.CANCELLED, cls.COMPLETED})


# Configuração de transições (fora da classe para evitar referência circular)
[Entity]Status.ALLOWED_TRANSITIONS = {
    [Entity]Status.DRAFT: frozenset({[Entity]Status.ACTIVE, [Entity]Status.CANCELLED}),
    [Entity]Status.ACTIVE: frozenset({[Entity]Status.SUSPENDED, [Entity]Status.COMPLETED}),
    [Entity]Status.SUSPENDED: frozenset({[Entity]Status.ACTIVE, [Entity]Status.CANCELLED}),
    [Entity]Status.CANCELLED: frozenset(),  # terminal
    [Entity]Status.COMPLETED: frozenset(),  # terminal
}
```

---

## Checklist

- [ ] `frozen=True` no dataclass
- [ ] Validação em `__post_init__`
- [ ] Exceções descritivas com valores inválidos
- [ ] Operações retornam novos objetos (nunca mutam `self`)
- [ ] `__str__` e `__repr__` implementados
- [ ] Zero imports de frameworks
- [ ] Testes: valores válidos, inválidos, operações, edge cases

---

## Instruções de Uso

```
@workspace #file:.github/copilot/templates/hexagonal/domain/value-object.md

Crie o Value Object [Nome] que representa [o que é no domínio].
Atributos: [lista]
Validações: [lista de regras]
Operações: [lista de operações que deve suportar]
```
