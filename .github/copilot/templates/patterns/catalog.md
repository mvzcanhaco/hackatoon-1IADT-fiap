# Catálogo de Patterns — Referência Rápida

Use este catálogo para identificar o pattern correto para cada situação e obter o template de implementação.

---

## Índice

| Pattern | Categoria | Use Quando |
|---------|-----------|------------|
| [Strategy](#strategy) | Comportamental | Múltiplos algoritmos intercambiáveis |
| [Factory Method](#factory-method) | Criacional | Criação polimórfica em runtime |
| [Abstract Factory](#abstract-factory) | Criacional | Famílias de objetos relacionados |
| [Builder](#builder) | Criacional | Construção passo-a-passo de objetos complexos |
| [Repository](#repository) | Arquitetural | Abstração de coleção de aggregates |
| [Adapter](#adapter) | Estrutural | Converter interface de terceiro |
| [Decorator](#decorator) | Estrutural | Adicionar comportamento sem herança |
| [Facade](#facade) | Estrutural | Simplificar subsistema complexo |
| [Observer](#observer) | Comportamental | Notificações de eventos |
| [Specification](#specification) | Domínio | Regras de seleção/filtro reutilizáveis |
| [Command](#command) | Comportamental | Encapsular operações como objetos |
| [Template Method](#template-method) | Comportamental | Algoritmo com passos variáveis |

---

## Strategy

**Problema**: Múltiplos algoritmos para a mesma operação. If/elif switch que cresce.

**Solução**: Extrair cada algoritmo em uma classe que implementa uma interface comum.

**Exemplo de uso neste projeto**:
```python
# Vários detectores (GPU, CPU) implementam a mesma interface
class DetectorInterface(ABC):
    @abstractmethod
    def process_video(self, ...) -> DetectionResult: ...

class GpuDetector(DetectorInterface): ...
class CpuDetector(DetectorInterface): ...
```

**Template**:
```python
from abc import ABC, abstractmethod

class [StrategyInterface](ABC):
    @abstractmethod
    def [execute](self, [params]: [types]) -> [ReturnType]: ...

class [ConcreteStrategy1]([StrategyInterface]):
    def [execute](self, [params]: [types]) -> [ReturnType]:
        # implementação 1
        ...

class [ConcreteStrategy2]([StrategyInterface]):
    def [execute](self, [params]: [types]) -> [ReturnType]:
        # implementação 2
        ...

class [Context]:
    def __init__(self, strategy: [StrategyInterface]) -> None:
        self._strategy = strategy

    def [operation](self, [params]) -> [ReturnType]:
        return self._strategy.[execute]([params])
```

---

## Factory Method

**Problema**: Decidir qual objeto concreto criar em runtime sem que o caller saiba.

**Template**:
```python
from abc import ABC, abstractmethod

class [Creator](ABC):
    @abstractmethod
    def create(self) -> [Product]:
        """Subclasses decidem qual produto concreto criar."""

    def [operation](self) -> [Result]:
        product = self.create()      # usa o produto sem saber a implementação
        return product.[do_something]()

class [ConcreteCreator1]([Creator]):
    def create(self) -> [Product]:
        return [ConcreteProduct1]()

class [ConcreteCreator2]([Creator]):
    def create(self) -> [Product]:
        return [ConcreteProduct2](config=special_config)
```

---

## Adapter

**Problema**: Interface de terceiro não bate com a interface esperada pelo domínio.

**Template**:
```python
# Port (interface do domínio)
class [DomainPort](ABC):
    @abstractmethod
    def [domain_operation](self, [domain_params]) -> [DomainReturn]: ...

# Adapter (infra — converte chamadas)
class [ThirdParty]Adapter([DomainPort]):
    def __init__(self, third_party_client: [ThirdPartyType]) -> None:
        self._client = third_party_client

    def [domain_operation](self, [domain_params]) -> [DomainReturn]:
        # Converte parâmetros do domínio → formato do terceiro
        third_party_request = self._convert_request([domain_params])
        # Chama o serviço externo
        third_party_response = self._client.[third_party_method](third_party_request)
        # Converte resposta do terceiro → formato do domínio
        return self._convert_response(third_party_response)

    def _convert_request(self, [domain_params]) -> [ThirdPartyRequest]: ...
    def _convert_response(self, response: [ThirdPartyResponse]) -> [DomainReturn]: ...
```

---

## Decorator

**Problema**: Adicionar comportamento (logging, cache, validação) sem modificar a classe.

**Template**:
```python
class [Logging|Cache|Retry|Metric]Decorator([Interface]):
    def __init__(self, wrapped: [Interface]) -> None:
        self._wrapped = wrapped

    def [method](self, [params]) -> [ReturnType]:
        # Comportamento ANTES
        self._before([params])

        try:
            result = self._wrapped.[method]([params])
            # Comportamento DEPOIS (sucesso)
            self._after_success(result)
            return result
        except Exception as e:
            # Comportamento DEPOIS (erro)
            self._after_error(e)
            raise

    def _before(self, [params]) -> None: ...
    def _after_success(self, result: [ReturnType]) -> None: ...
    def _after_error(self, error: Exception) -> None: ...
```

---

## Observer / Event Dispatcher

**Problema**: Notificar múltiplos sistemas quando um evento de domínio ocorre.

**Template**:
```python
from typing import Callable, Dict, List, Type

class DomainEventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[Type, List[Callable]] = {}

    def subscribe(self, event_type: Type, handler: Callable) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event: object) -> None:
        for handler in self._subscribers.get(type(event), []):
            handler(event)

# Registro de handlers:
bus = DomainEventBus()
bus.subscribe(ThreatDetectedEvent, send_email_notification)
bus.subscribe(ThreatDetectedEvent, trigger_webhook)
bus.subscribe(ThreatDetectedEvent, update_dashboard_metrics)
```

---

## Specification

**Problema**: Regras de seleção/filtro espalhadas por vários lugares, difíceis de combinar.

**Template**:
```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")

class Specification(ABC, Generic[T]):
    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool: ...

    def and_(self, other: "Specification[T]") -> "AndSpec[T]":
        return AndSpec(self, other)

    def or_(self, other: "Specification[T]") -> "OrSpec[T]":
        return OrSpec(self, other)

    def not_(self) -> "NotSpec[T]":
        return NotSpec(self)

class AndSpec(Specification[T]):
    def __init__(self, left: Specification[T], right: Specification[T]):
        self._left, self._right = left, right
    def is_satisfied_by(self, c: T) -> bool:
        return self._left.is_satisfied_by(c) and self._right.is_satisfied_by(c)

# Especificações concretas:
class HighConfidenceSpec(Specification[Detection]):
    def __init__(self, min_confidence: float = 0.8):
        self._min = min_confidence
    def is_satisfied_by(self, d: Detection) -> bool:
        return d.confidence >= self._min

class WeaponLabelSpec(Specification[Detection]):
    def __init__(self, label: str):
        self._label = label
    def is_satisfied_by(self, d: Detection) -> bool:
        return self._label.lower() in d.label.lower()

# Composição:
critical_threat = HighConfidenceSpec(0.9).and_(WeaponLabelSpec("knife"))
results = [d for d in detections if critical_threat.is_satisfied_by(d)]
```

---

## Repository

Ver template completo em: `.github/copilot/templates/hexagonal/infrastructure/repository.md`

---

## Template Method

**Problema**: Algoritmo com estrutura fixa mas passos variáveis.

**Template**:
```python
class [AlgorithmBase](ABC):
    def execute(self, [params]) -> [ReturnType]:
        """Template method — define a estrutura do algoritmo."""
        data = self._step_1_prepare([params])
        processed = self._step_2_process(data)
        result = self._step_3_format(processed)
        self._step_4_cleanup()
        return result

    @abstractmethod
    def _step_1_prepare(self, [params]) -> [IntermediateType]: ...

    @abstractmethod
    def _step_2_process(self, data: [IntermediateType]) -> [ProcessedType]: ...

    def _step_3_format(self, processed: [ProcessedType]) -> [ReturnType]:
        """Hook com implementação padrão — pode ser sobrescrito."""
        return processed

    def _step_4_cleanup(self) -> None:
        """Hook opcional — subclasses sobrescrevem se necessário."""
        pass
```

---

## Como Usar Este Catálogo

```
@workspace #file:.github/copilot/templates/patterns/catalog.md
#file:.github/copilot/skills/design-patterns.md

Preciso implementar [descreva o problema].
Qual pattern do catálogo se aplica? Me dê o código adaptado para o contexto do projeto.
```
