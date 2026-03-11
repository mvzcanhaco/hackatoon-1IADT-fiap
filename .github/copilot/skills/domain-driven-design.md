# Skill: Domain-Driven Design (DDD)

## O que é

DDD (Eric Evans, 2003) é uma abordagem de design de software que coloca o domínio do negócio no centro das decisões técnicas. O código deve espelhar o modelo mental dos especialistas de domínio.

**Princípio Central**: O software deve falar a linguagem do negócio, não o contrário.

---

## Ubiquitous Language

A linguagem ubíqua é o vocabulário compartilhado entre desenvolvedores e especialistas de domínio. **Todo código deve usar os termos exatos do domínio**.

```python
# ❌ Linguagem técnica / genérica
class DataProcessor:
    def process(self, input_data: dict) -> dict:
        ...

# ✅ Linguagem do domínio
class WeaponDetector:
    def analyze_video_frame(self, frame: VideoFrame) -> DetectionResult:
        ...
```

### Como construir o glossário:
1. Entreviste especialistas de domínio
2. Liste todos os substantivos (entidades, VOs, coleções)
3. Liste todos os verbos (operações, eventos)
4. Elimine sinônimos — escolha **um** termo para cada conceito
5. Documente em `docs/ubiquitous-language.md`

---

## Bounded Contexts

Um Bounded Context é uma fronteira explícita dentro da qual um modelo de domínio é consistente e tem significado preciso.

**O mesmo termo pode ter significados diferentes em contextos distintos:**

```
Context: Vendas          Context: Logística        Context: Faturamento
─────────────────        ──────────────────        ───────────────────
"Pedido":                "Pedido":                 "Pedido":
- Itens solicitados      - Carga para entrega      - Documento fiscal
- Preço e desconto       - Endereço e prazo        - CNPJ e impostos
- Status de pagamento    - Status de transporte    - Status de emissão
```

### Relações entre Bounded Contexts

| Relação | Descrição | Quando usar |
|---------|-----------|-------------|
| **Shared Kernel** | Modelo compartilhado entre contexts | Equipes muito alinhadas |
| **Customer-Supplier** | Um context serve outro (contrato) | Dependência clara de um para o outro |
| **Conformist** | Downstream aceita modelo do upstream sem negociar | Contexto externo (API de terceiro) |
| **Anti-Corruption Layer (ACL)** | Traduz modelos entre contexts | Integração com legado ou serviço externo |
| **Open Host Service** | Expõe protocolo para múltiplos consumers | Serviço interno de plataforma |

---

## Blocos Construtivos do DDD

### Entity (Entidade)

Objeto com **identidade única** que persiste no tempo, mesmo que seus atributos mudem.

**Critério**: Dois objetos com os mesmos atributos mas IDs diferentes são **diferentes**.

```python
@dataclass
class Customer:
    id: CustomerId          # identidade imutável
    name: str               # pode mudar
    email: Email            # pode mudar

    def change_email(self, new_email: Email) -> "EmailChangedEvent":
        """Regra: email deve ser verificado antes de ser trocado."""
        if new_email == self.email:
            raise SameEmailError()
        self.email = new_email
        return EmailChangedEvent(customer_id=self.id, new_email=new_email)
```

---

### Value Object (Objeto de Valor)

Objeto **sem identidade** descrito apenas por seus atributos. **Imutável**.

**Critério**: Dois objetos com os mesmos atributos são **iguais** (como números).

```python
@dataclass(frozen=True)  # frozen=True garante imutabilidade
class Email:
    value: str

    def __post_init__(self) -> None:
        if "@" not in self.value or "." not in self.value.split("@")[-1]:
            raise InvalidEmailError(self.value)

    def domain(self) -> str:
        return self.value.split("@")[1]

    def __str__(self) -> str:
        return self.value

# Uso:
email1 = Email("user@example.com")
email2 = Email("user@example.com")
assert email1 == email2  # True — igualdade por valor, não por referência
```

**Candidatos a Value Object**:
- CPF, CNPJ, Email, Telefone
- Money (valor + moeda)
- Address (logradouro, cidade, CEP)
- DateRange (início + fim)
- Percentage, Temperature, Coordinate

---

### Aggregate

Cluster de entidades e VOs tratados como **uma unidade de consistência**.

```
┌─────────────────────────────────────────┐
│  Aggregate: Order                       │
│                                         │
│  ┌───────────────┐                      │
│  │ Order         │ ← Aggregate Root     │
│  │ (id, status)  │   (único ponto de    │
│  └───────┬───────┘    acesso externo)   │
│          │                              │
│  ┌───────▼───────┐  ┌──────────────┐   │
│  │  OrderItem    │  │  OrderItem   │   │
│  │  (product,    │  │  (product,   │   │
│  │   quantity,   │  │   quantity,  │   │
│  │   price)      │  │   price)     │   │
│  └───────────────┘  └──────────────┘   │
└─────────────────────────────────────────┘
      ↑
   Apenas através do Aggregate Root
   (ninguém acessa OrderItem diretamente)
```

**Regras de Aggregate**:
1. Acesso externo apenas via Aggregate Root
2. Invariantes garantidas dentro do aggregate
3. Referências entre aggregates apenas por ID (não por objeto)
4. Um aggregate = uma transação de banco

```python
class Order:  # Aggregate Root
    def __init__(self, ...):
        self._items: List[OrderItem] = []  # privado

    def add_item(self, product_id: ProductId, quantity: int, unit_price: Money) -> None:
        """Acesso controlado — garante invariantes."""
        if self._status != OrderStatus.DRAFT:
            raise OrderNotEditableError(self.id)
        if quantity <= 0:
            raise InvalidQuantityError(quantity)
        self._items.append(OrderItem(product_id, quantity, unit_price))

    @property
    def total(self) -> Money:
        return sum(item.subtotal for item in self._items)

    @property
    def items(self) -> tuple:
        return tuple(self._items)  # imutável para o exterior
```

---

### Repository (Repositório)

Abstração de coleção de aggregates. **A interface é no domínio, a implementação na infra**.

```python
# No domínio — interface pura
from abc import ABC, abstractmethod
from typing import Optional, List
from ..entities.order import Order
from ..value_objects.order_id import OrderId

class OrderRepository(ABC):
    """Port de persistência para o aggregate Order."""

    @abstractmethod
    def find_by_id(self, order_id: OrderId) -> Optional[Order]:
        ...

    @abstractmethod
    def find_by_customer(self, customer_id: CustomerId) -> List[Order]:
        ...

    @abstractmethod
    def save(self, order: Order) -> None:
        ...

    @abstractmethod
    def delete(self, order_id: OrderId) -> None:
        ...
```

---

### Domain Service

Lógica de domínio que **não pertence naturalmente a nenhuma entidade ou VO**.

**Use quando**: a operação envolve múltiplos aggregates ou requer dados que a entidade não possui.

```python
# Exemplo: calcular frete requer Order + DeliveryZone + PriceTable
class FreightCalculator(ABC):
    """Domain Service — calcula o custo de frete."""

    @abstractmethod
    def calculate(
        self,
        order: Order,
        destination: Address,
        carrier: Carrier,
    ) -> Money:
        """
        Regra de negócio: frete depende do peso total,
        distância e transportadora selecionada.
        """
```

---

### Domain Events (Eventos de Domínio)

Representam **algo que aconteceu** no domínio. Imutáveis, nomeados no passado.

```python
@dataclass(frozen=True)
class OrderConfirmed:
    """Evento: Pedido foi confirmado pelo cliente."""
    order_id: OrderId
    customer_id: CustomerId
    total: Money
    confirmed_at: datetime = field(default_factory=datetime.utcnow)
    event_id: UUID = field(default_factory=uuid4)

# Produzido pela entidade:
class Order:
    def confirm(self) -> OrderConfirmed:
        if not self._items:
            raise EmptyOrderError()
        self._status = OrderStatus.CONFIRMED
        return OrderConfirmed(
            order_id=self.id,
            customer_id=self._customer_id,
            total=self.total,
        )
```

---

## Context Map — Template

```markdown
## Context Map: [Nome do Sistema]

### Contexts Identificados:
1. **[Context A]** — responsabilidade: ...
2. **[Context B]** — responsabilidade: ...
3. **[Context C]** — responsabilidade: ...

### Relações:
- [Context A] → Customer-Supplier → [Context B]
  (A fornece dados de produto para B processar pedidos)

- [Context B] → ACL → [Context C externo / legado]
  (B traduz modelo externo de pagamento para o modelo interno)

### Shared Kernel:
- [Definir tipos compartilhados se houver]
```

---

## Quando Usar Este Skill

- Nomear classes, métodos e variáveis (Ubiquitous Language)
- Decidir se um objeto é Entity, VO, ou Domain Service
- Definir fronteiras de Aggregates
- Mapear Bounded Contexts e suas relações
- Identificar onde colocar eventos de domínio
