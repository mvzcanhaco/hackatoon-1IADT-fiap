# Skill: Pirâmide de Testes

## Estratégia de Testes por Camada

```
              ▲
             /E2E\          ~5%   — Fluxos críticos completos
            /─────\
           / Integ.\        ~20%  — Repositórios, Adapters, APIs
          /─────────\
         /  Unitários \     ~75%  — Domínio, Use Cases, Lógica pura
        /───────────────\
```

---

## Tipos de Test Doubles

| Tipo | O que faz | Quando usar |
|------|-----------|-------------|
| **Dummy** | Objeto vazio que nunca é chamado | Preencher parâmetros obrigatórios irrelevantes |
| **Stub** | Retorna valores pré-definidos | Simular dados de repositório em happy path |
| **Fake** | Implementação simplificada real | Substituir banco com lista em memória |
| **Mock** | Verifica que foi chamado corretamente | Verificar que `save()` foi chamado |
| **Spy** | Registra chamadas para verificação posterior | Verificar notificações enviadas |

```python
# Stub — retorna dados fixos
mock_repo = Mock(spec=OrderRepository)
mock_repo.find_by_id.return_value = order_fixture

# Mock — verifica que foi chamado
mock_repo.save.assert_called_once_with(expected_order)
mock_repo.save.assert_called_once()
mock_repo.save.assert_not_called()

# Fake — implementação real em memória
class InMemoryOrderRepository(OrderRepository):
    def __init__(self):
        self._store: Dict[str, Order] = {}

    def find_by_id(self, order_id: OrderId) -> Optional[Order]:
        return self._store.get(str(order_id))

    def save(self, order: Order) -> None:
        self._store[str(order.id)] = order
```

---

## Estrutura de Arquivo de Teste

```python
"""
Testes para [NomeDoMódulo].

Princípios:
- Um conceito por teste
- Arrange / Act / Assert (AAA)
- Nome do teste = documentação do comportamento
"""
import pytest
from unittest.mock import Mock

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_order() -> Order:
    """Pedido em estado válido (DRAFT) com um item."""
    return OrderBuilder().with_item("product-1", 2).build()


@pytest.fixture
def confirmed_order() -> Order:
    """Pedido já confirmado — operações de edição devem falhar."""
    return OrderBuilder().with_item("product-1", 2).in_confirmed_state().build()


# ─── Happy Path Tests ────────────────────────────────────────────────────────

class TestOrderConfirmation:
    """Comportamento esperado ao confirmar um pedido."""

    def test_confirm_changes_status_to_confirmed(self, valid_order: Order) -> None:
        """Pedido com itens deve mudar status para CONFIRMED."""
        valid_order.confirm()
        assert valid_order.status == OrderStatus.CONFIRMED

    def test_confirm_emits_order_confirmed_event(self, valid_order: Order) -> None:
        """Confirmação deve emitir evento OrderConfirmed."""
        event = valid_order.confirm()
        assert isinstance(event, OrderConfirmed)
        assert event.order_id == valid_order.id

    def test_confirm_preserves_order_items(self, valid_order: Order) -> None:
        """Confirmação não deve modificar os itens do pedido."""
        original_items = list(valid_order.items)
        valid_order.confirm()
        assert list(valid_order.items) == original_items


# ─── Error / Invariant Tests ──────────────────────────────────────────────────

class TestOrderConfirmationErrors:
    """Invariantes e casos de erro na confirmação."""

    def test_confirm_raises_when_order_is_empty(self) -> None:
        """Pedido sem itens não pode ser confirmado."""
        empty_order = OrderBuilder().build()
        with pytest.raises(EmptyOrderError):
            empty_order.confirm()

    def test_confirm_raises_when_already_confirmed(self, confirmed_order: Order) -> None:
        """Pedido já confirmado não pode ser confirmado novamente."""
        with pytest.raises(OrderAlreadyConfirmedError):
            confirmed_order.confirm()

    def test_add_item_raises_when_order_is_confirmed(self, confirmed_order: Order) -> None:
        """Não deve ser possível adicionar itens a pedido confirmado."""
        with pytest.raises(OrderNotEditableError):
            confirmed_order.add_item(ProductId("p1"), quantity=1, unit_price=Money(10, "BRL"))


# ─── Parametrized Tests ──────────────────────────────────────────────────────

class TestOrderTotal:
    """Cálculo do total do pedido."""

    @pytest.mark.parametrize("items,expected_total", [
        ([(1, 10.00)], 10.00),
        ([(2, 10.00)], 20.00),
        ([(1, 10.00), (3, 5.00)], 25.00),
        ([(10, 0.01)], 0.10),
    ])
    def test_total_calculates_correctly(
        self,
        items: List[Tuple[int, float]],
        expected_total: float,
    ) -> None:
        """Total do pedido deve ser soma de (quantidade × preço unitário) de todos os itens."""
        order = OrderBuilder()
        for quantity, price in items:
            order.with_item("product", quantity, unit_price=price)
        assert order.build().total == Money(expected_total, "BRL")
```

---

## Padrões de Nomenclatura para Testes

```
test_[método]_[resultado_esperado]_when_[condição]
test_[método]_returns_[valor]_given_[contexto]
test_[método]_raises_[exceção]_when_[condição]
test_[comportamento]_[resultado]

# Exemplos:
test_confirm_raises_empty_order_error_when_no_items
test_add_item_increases_order_total
test_process_video_sends_notification_when_threat_detected
test_process_video_does_not_send_notification_when_safe
```

---

## Configuração pytest — `conftest.py`

```python
# tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.infrastructure.config.database import Base


@pytest.fixture(scope="session")
def db_engine():
    """Engine SQLite em memória para toda a sessão de testes."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Session isolada por teste — rollback automático ao final."""
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def mock_notification_factory() -> Mock:
    """Factory de notificação mockada para testes de use case."""
    factory = Mock(spec=NotificationFactory)
    factory.create_service.return_value = Mock(spec=NotificationInterface)
    return factory
```

---

## Cobertura — O que medir

**Não otimize para % de linhas — otimize para % de comportamentos de negócio.**

```ini
# pytest.ini ou pyproject.toml
[pytest]
addopts = --cov=src --cov-report=html --cov-fail-under=80

# .coveragerc
[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if TYPE_CHECKING:
    raise NotImplementedError
    pass
```

### Cobertura por camada:

| Camada | Meta de Cobertura | Tipo de Teste |
|--------|-------------------|---------------|
| `domain/` | 95%+ | Unitário |
| `application/use_cases/` | 90%+ | Unitário (com mocks) |
| `infrastructure/repositories/` | 80%+ | Integração |
| `presentation/` | 70%+ | Unitário + E2E |

---

## Anti-patterns em Testes

### ❌ Teste que testa a implementação, não o comportamento
```python
# Ruim — testa detalhe interno, não comportamento
def test_order_items_list_has_correct_type():
    order = Order(...)
    assert isinstance(order._items, list)  # testa implementação interna
```

### ✅ Testa comportamento observável
```python
# Bom — testa comportamento externo
def test_order_with_added_item_has_one_item():
    order = OrderBuilder().build()
    order.add_item(ProductId("p1"), quantity=1, unit_price=Money(10, "BRL"))
    assert len(order.items) == 1
```

### ❌ Teste gigante que verifica tudo de uma vez
```python
def test_full_order_flow():
    # 50 linhas verificando tudo
    # quando falhar, onde está o problema?
```

### ✅ Um conceito por teste, nome descritivo
```python
def test_add_item_increases_item_count_by_one(): ...
def test_add_item_increases_order_total(): ...
def test_confirm_changes_status(): ...
```

---

## Quando Usar Este Skill

```
@workspace #file:.github/copilot/skills/testing-pyramid.md
#file:src/domain/entities/[entity].py

Gere os testes para esta entidade seguindo a estratégia de pirâmide de testes.
Cubra: [lista de comportamentos a testar]
```
