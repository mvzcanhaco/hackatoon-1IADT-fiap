# Perfil de Linguagem: Java / Spring Boot

## Referências Canônicas

- **Effective Java** (Bloch): https://www.oreilly.com/library/view/effective-java/9780134686097/
- **Spring Boot Reference**: https://docs.spring.io/spring-boot/docs/current/reference/html/
- **Google Java Style Guide**: https://google.github.io/styleguide/javaguide.html
- **Java 21 JEPs**: https://openjdk.org/projects/jdk/21/

---

## Detecção Automática

Este perfil é ativado quando o workspace contém:
- `pom.xml` ou `build.gradle` / `build.gradle.kts`
- Arquivos `.java`
- `src/main/java/` (estrutura Maven/Gradle)

**Verificar versão obrigatoriamente**:
```bash
cat pom.xml | grep '<java.version>'     # Ex: <java.version>21</java.version>
# ou
cat build.gradle.kts | grep jvmTarget   # Ex: jvmTarget = "21"
java --version                           # Ex: openjdk 21.0.2
```

---

## Versão Mínima: Java 21 LTS

Features obrigatórias por versão:

| Feature | Versão | Uso |
|---------|--------|-----|
| Records | Java 16+ | Value Objects e DTOs imutáveis |
| Sealed classes | Java 17+ | Hierarquia de exceções fechada |
| Pattern matching `instanceof` | Java 16+ | Substituir casts explícitos |
| Text blocks | Java 15+ | SQL, JSON inline sem escapes |
| Switch expressions | Java 14+ | Substituir switch statements |
| Virtual threads (Project Loom) | Java 21+ | Alta concorrência sem reactive |

---

## Convenções de Nomenclatura

| Elemento | Convenção | Exemplo |
|----------|-----------|---------|
| Classe / Interface | `PascalCase` | `OrderRepository`, `ProcessOrderUseCase` |
| Método | `camelCase` | `findById()`, `processOrder()` |
| Variável / Parâmetro | `camelCase` | `orderId`, `customerEmail` |
| Constante (`static final`) | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_PAGE_SIZE` |
| Pacote | `lowercase.separated.by.dots` | `com.company.domain.entities` |
| Interface (Port) | `*Repository`, `*Gateway`, `*Port` | `OrderRepository`, `EmailGateway` |
| Record (DTO/VO) | `PascalCase` | `CreateOrderRequest`, `Money` |
| Enum | `PascalCase` + valores `UPPER_SNAKE` | `OrderStatus.CONFIRMED` |

---

## Estrutura de Pacotes (Hexagonal)

```
src/main/java/com/[company]/[context]/
├── domain/
│   ├── model/              ← Entidades e Value Objects (POJOs puros — zero Spring)
│   │   ├── Order.java
│   │   └── Money.java
│   ├── repository/         ← Interfaces de repositório (ports secundários)
│   │   └── OrderRepository.java
│   ├── port/               ← Outros ports (Gateway, Publisher, etc.)
│   │   └── EmailGateway.java
│   └── exception/          ← Exceções de domínio
│       └── OrderNotFoundException.java
├── application/
│   ├── usecase/            ← Use cases — orquestração pura
│   │   └── ProcessOrderUseCase.java
│   └── dto/                ← DTOs de entrada/saída dos use cases
│       ├── ProcessOrderRequest.java
│       └── ProcessOrderResponse.java
├── infrastructure/
│   ├── persistence/        ← Implementações JPA dos repositórios
│   │   ├── JpaOrderRepository.java
│   │   └── OrderJpaEntity.java
│   ├── gateway/            ← Implementações de Gateways externos
│   │   └── SendGridEmailGateway.java
│   └── config/             ← Configuração Spring (@Configuration)
│       └── UseCaseConfig.java
└── presentation/
    ├── http/               ← Controllers REST
    │   ├── OrderController.java
    │   └── dto/            ← Request/Response schemas HTTP
    │       ├── CreateOrderHttpRequest.java
    │       └── OrderHttpResponse.java
    └── advice/             ← Exception handlers globais
        └── GlobalExceptionHandler.java
```

**Regra absoluta**: `domain/` nunca importa nada de `org.springframework.*` nem de JPA.

---

## Entidade de Domínio (Java Puro)

```java
// src/main/java/com/[company]/[context]/domain/model/Order.java
package com.company.context.domain.model;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

public final class Order {

    private final OrderId id;
    private final CustomerId customerId;
    private final List<OrderItem> items;
    private OrderStatus status;
    private final Instant createdAt;

    private Order(OrderId id, CustomerId customerId, Instant createdAt) {
        this.id = id;
        this.customerId = customerId;
        this.items = new ArrayList<>();
        this.status = OrderStatus.DRAFT;
        this.createdAt = createdAt;
    }

    public static Order create(CustomerId customerId) {
        return new Order(OrderId.generate(), customerId, Instant.now());
    }

    public void addItem(ProductId productId, int quantity, Money unitPrice) {
        if (status != OrderStatus.DRAFT) {
            throw new OrderNotEditableException(id, status);
        }
        if (quantity <= 0) {
            throw new InvalidQuantityException(quantity);
        }
        items.add(new OrderItem(productId, quantity, unitPrice));
    }

    public void confirm() {
        if (status != OrderStatus.DRAFT) {
            throw new InvalidStateTransitionException("confirm", status);
        }
        if (items.isEmpty()) {
            throw new EmptyOrderException(id);
        }
        this.status = OrderStatus.CONFIRMED;
    }

    // Getters — sem setters públicos
    public OrderId getId() { return id; }
    public OrderStatus getStatus() { return status; }
    public List<OrderItem> getItems() { return Collections.unmodifiableList(items); }
    public Instant getCreatedAt() { return createdAt; }
}
```

---

## Value Object com Record (Java 16+)

```java
// Value Objects imutáveis usando Records
public record Money(BigDecimal amount, String currency) {

    public Money {
        Objects.requireNonNull(amount, "amount must not be null");
        Objects.requireNonNull(currency, "currency must not be null");
        if (amount.compareTo(BigDecimal.ZERO) < 0) {
            throw new InvalidValueException("Money amount cannot be negative: " + amount);
        }
        if (currency.isBlank() || currency.length() != 3) {
            throw new InvalidValueException("Invalid currency code: " + currency);
        }
    }

    public Money add(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new CurrencyMismatchException(this.currency, other.currency);
        }
        return new Money(this.amount.add(other.amount), this.currency);
    }

    public static Money of(String amount, String currency) {
        return new Money(new BigDecimal(amount), currency);
    }
}

// ID tipado como Record (evita confusão entre IDs)
public record OrderId(String value) {

    public OrderId {
        Objects.requireNonNull(value, "OrderId value must not be null");
        if (value.isBlank()) throw new InvalidValueException("OrderId cannot be blank");
    }

    public static OrderId generate() {
        return new OrderId(UUID.randomUUID().toString());
    }

    public static OrderId of(String value) {
        return new OrderId(value);
    }

    @Override public String toString() { return value; }
}
```

---

## Port Interface (Repository)

```java
// src/main/java/com/[company]/[context]/domain/repository/OrderRepository.java
package com.company.context.domain.repository;

import java.util.List;
import java.util.Optional;

// Port puro — zero Spring, zero JPA
public interface OrderRepository {
    Optional<Order> findById(OrderId id);
    List<Order> findByCustomer(CustomerId customerId, int page, int pageSize);
    boolean existsById(OrderId id);
    void save(Order order);
    void delete(OrderId id);
}
```

---

## Use Case

```java
// src/main/java/com/[company]/[context]/application/usecase/ProcessOrderUseCase.java
package com.company.context.application.usecase;

import lombok.extern.slf4j.Slf4j;

@Slf4j  // apenas na aplicação/infra — nunca no domínio
public class ProcessOrderUseCase {

    private final OrderRepository orderRepository;
    private final PaymentGateway paymentGateway;

    // Injeção via construtor — não @Autowired no campo
    public ProcessOrderUseCase(OrderRepository orderRepository, PaymentGateway paymentGateway) {
        this.orderRepository = orderRepository;
        this.paymentGateway = paymentGateway;
    }

    public ProcessOrderResponse execute(ProcessOrderRequest request) {
        log.info("Processing order: orderId={}", request.orderId());

        Order order = orderRepository.findById(OrderId.of(request.orderId()))
            .orElseThrow(() -> new OrderNotFoundException(request.orderId()));

        order.confirm();

        PaymentResult payment = paymentGateway.charge(order.total(), request.paymentMethod());

        orderRepository.save(order);

        log.info("Order processed successfully: orderId={}", request.orderId());
        return new ProcessOrderResponse(order.getId().value(), order.getStatus().name());
    }
}

// DTOs como Records
public record ProcessOrderRequest(String orderId, String paymentMethod) {}
public record ProcessOrderResponse(String orderId, String status) {}
```

---

## Implementação JPA do Repository

```java
// src/main/java/com/[company]/[context]/infrastructure/persistence/JpaOrderRepository.java
@Repository
public class JpaOrderRepository implements OrderRepository {

    private final SpringDataOrderRepository springRepo;

    public JpaOrderRepository(SpringDataOrderRepository springRepo) {
        this.springRepo = springRepo;
    }

    @Override
    public Optional<Order> findById(OrderId id) {
        return springRepo.findById(id.value()).map(this::toDomain);
    }

    @Override
    public void save(Order order) {
        springRepo.save(toEntity(order));
    }

    private Order toDomain(OrderEntity entity) {
        // Mapeamento: entity → domain (sem framework no domínio)
        return Order.reconstitute(  // factory method para reconstrução
            OrderId.of(entity.getId()),
            CustomerId.of(entity.getCustomerId()),
            OrderStatus.valueOf(entity.getStatus()),
            entity.getCreatedAt()
        );
    }

    private OrderEntity toEntity(Order order) {
        OrderEntity entity = new OrderEntity();
        entity.setId(order.getId().value());
        entity.setStatus(order.getStatus().name());
        return entity;
    }
}

// Interface Spring Data (somente na infra)
interface SpringDataOrderRepository extends JpaRepository<OrderEntity, String> {
    List<OrderEntity> findByCustomerIdOrderByCreatedAtDesc(String customerId, Pageable pageable);
}
```

---

## Controller REST (Spring MVC)

```java
// src/main/java/com/[company]/[context]/presentation/http/OrderController.java
@RestController
@RequestMapping("/api/v1/orders")
@Validated
public class OrderController {

    private final ProcessOrderUseCase processOrderUseCase;

    public OrderController(ProcessOrderUseCase processOrderUseCase) {
        this.processOrderUseCase = processOrderUseCase;
    }

    @PostMapping("/{orderId}/process")
    public ResponseEntity<OrderHttpResponse> process(
        @PathVariable String orderId,
        @RequestBody @Valid ProcessOrderHttpRequest request
    ) {
        ProcessOrderResponse result = processOrderUseCase.execute(
            new ProcessOrderRequest(orderId, request.paymentMethod())
        );
        return ResponseEntity.ok(new OrderHttpResponse(result.orderId(), result.status()));
    }
}

// Exception handler global
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(OrderNotFoundException.class)
    public ResponseEntity<ErrorResponse> handle(OrderNotFoundException ex) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND)
            .body(new ErrorResponse(ex.getMessage(), "ORDER_NOT_FOUND"));
    }

    @ExceptionHandler(InvalidStateTransitionException.class)
    public ResponseEntity<ErrorResponse> handle(InvalidStateTransitionException ex) {
        return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY)
            .body(new ErrorResponse(ex.getMessage(), "INVALID_STATE_TRANSITION"));
    }
}
```

---

## Configuração de DI (Spring)

```java
// src/main/java/com/[company]/[context]/infrastructure/config/UseCaseConfig.java
@Configuration
public class UseCaseConfig {

    // Monta o grafo de dependências explicitamente
    @Bean
    public ProcessOrderUseCase processOrderUseCase(
        OrderRepository orderRepository,
        PaymentGateway paymentGateway
    ) {
        return new ProcessOrderUseCase(orderRepository, paymentGateway);
    }
}
```

---

## Testes

### Unitário — Entidade de Domínio (JUnit 5 + AssertJ)

```java
class OrderTest {

    @Test
    void confirm_shouldTransitionToDraftWhenItemsPresent() {
        Order order = Order.create(CustomerId.of("c1"));
        order.addItem(ProductId.of("p1"), 2, Money.of("10.00", "BRL"));

        order.confirm();

        assertThat(order.getStatus()).isEqualTo(OrderStatus.CONFIRMED);
    }

    @Test
    void confirm_shouldThrowWhenOrderIsEmpty() {
        Order order = Order.create(CustomerId.of("c1"));

        assertThatThrownBy(order::confirm)
            .isInstanceOf(EmptyOrderException.class)
            .hasMessageContaining(order.getId().value());
    }

    @Test
    void addItem_shouldThrowWhenOrderAlreadyConfirmed() {
        Order order = OrderBuilder.aConfirmedOrder().build();

        assertThatThrownBy(() ->
            order.addItem(ProductId.of("p2"), 1, Money.of("5.00", "BRL"))
        ).isInstanceOf(OrderNotEditableException.class);
    }
}
```

### Integração — Repositório (Testcontainers)

```java
@SpringBootTest
@Testcontainers
class JpaOrderRepositoryTest {

    @Container
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:16-alpine");

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", postgres::getJdbcUrl);
        registry.add("spring.datasource.username", postgres::getUsername);
        registry.add("spring.datasource.password", postgres::getPassword);
    }

    @Autowired JpaOrderRepository repository;

    @Test
    void save_andFindById_shouldRoundtrip() {
        Order order = Order.create(CustomerId.of("customer-1"));
        order.addItem(ProductId.of("p1"), 1, Money.of("99.90", "BRL"));

        repository.save(order);
        Optional<Order> found = repository.findById(order.getId());

        assertThat(found).isPresent();
        assertThat(found.get().getId()).isEqualTo(order.getId());
    }
}
```

---

## Limites de Qualidade

| Métrica | Limite |
|---------|--------|
| Máx. linhas/método | 30 |
| Máx. linhas/classe | 400 |
| Cobertura domínio | ≥ 90% |
| Cobertura aplicação | ≥ 85% |
| Linter | Checkstyle + SpotBugs + PMD |
| Formatter | google-java-format |

---

## Checklist de Qualidade

- [ ] `domain/` sem imports de `org.springframework.*` ou `jakarta.persistence.*`
- [ ] Value Objects como Records com validação no compact constructor
- [ ] Use Cases recebem dependências via construtor (não `@Autowired` em campo)
- [ ] Mapeamento domínio ↔ JPA entity nos repositórios de infraestrutura
- [ ] `@RestControllerAdvice` mapeia exceções de domínio → HTTP status
- [ ] Testes de domínio em JUnit 5 puro (sem Spring context)
- [ ] Testes de integração com Testcontainers (não H2)
- [ ] Records para DTOs e Value Objects imutáveis
- [ ] Sealed classes para hierarquias fechadas de exceções
