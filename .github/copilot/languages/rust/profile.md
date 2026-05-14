# Perfil de Linguagem: Rust

## Referências Canônicas

- **The Rust Book**: https://doc.rust-lang.org/book/
- **Rust API Guidelines**: https://rust-lang.github.io/api-guidelines/
- **Rust Async Book**: https://rust-lang.github.io/async-book/
- **Clippy Lints**: https://rust-lang.github.io/rust-clippy/
- **Tokio Tutorial**: https://tokio.rs/tokio/tutorial

---

## Detecção Automática

Este perfil é ativado quando o workspace contém:
- `Cargo.toml`
- Arquivos `.rs`
- `Cargo.lock`

**Verificar versão obrigatoriamente**:
```bash
cat Cargo.toml | grep 'edition'   # Ex: edition = "2021"
rustc --version                   # Ex: rustc 1.78.0 (9b00956e5 2024-04-29)
cargo --version                   # Ex: cargo 1.78.0
```

---

## Configuração: Cargo.toml

```toml
[package]
name = "[project-name]"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"  # MSRV (Minimum Supported Rust Version)
authors = ["[Nome] <[email]>"]
description = "[Descrição]"

[dependencies]
# HTTP framework
axum = { version = "0.7", features = ["macros"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Async runtime
tokio = { version = "1", features = ["full"] }

# Serialização
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Banco de dados
sqlx = { version = "0.7", features = ["runtime-tokio", "postgres", "uuid", "chrono"] }

# Error handling
thiserror = "1"     # para bibliotecas/domínio
anyhow = "1"        # para binários/aplicação (opcional)

# Utilitários
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
axum-test = "14"
tokio-test = "0.4"
mockall = "0.12"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
```

---

## Convenções de Nomenclatura

| Elemento | Convenção | Exemplo |
|----------|-----------|---------|
| Módulo / Arquivo | `snake_case` | `order_repository.rs`, `process_order.rs` |
| Struct / Enum / Trait | `PascalCase` | `OrderRepository`, `DomainError` |
| Função / Método | `snake_case` | `find_by_id()`, `process_order()` |
| Variável / Parâmetro | `snake_case` | `order_id`, `customer_email` |
| Constante | `UPPER_SNAKE_CASE` | `MAX_RETRIES`, `DEFAULT_TIMEOUT_MS` |
| Lifetime | `'a`, `'static`, `'req` | — letras curtas e descritivas |
| Trait (Port) | `*Repository`, `*Gateway`, `*Notifier` | `OrderRepository`, `EmailGateway` |
| Erro de domínio | `*Error` com `thiserror` | `OrderNotFoundError`, `InvalidStateError` |

---

## Estrutura de Módulos (Hexagonal)

```
src/
├── main.rs                  ← Entry point: tokio runtime, router, DI
├── domain/
│   ├── mod.rs
│   ├── entities/
│   │   ├── mod.rs
│   │   └── order.rs         ← Struct com comportamento + invariantes
│   ├── value_objects/
│   │   ├── mod.rs
│   │   ├── order_id.rs      ← Newtype pattern
│   │   └── money.rs
│   ├── repositories/        ← Traits (ports secundários)
│   │   └── order_repository.rs
│   ├── services/            ← Domain services
│   └── errors.rs            ← DomainError enum com thiserror
├── application/
│   ├── mod.rs
│   ├── use_cases/
│   │   └── process_order.rs ← Struct com execute() async
│   └── dtos/
│       └── process_order.rs ← Request/Response structs
├── infrastructure/
│   ├── mod.rs
│   ├── repositories/
│   │   └── sqlx_order_repository.rs ← impl OrderRepository
│   ├── gateways/
│   │   └── sendgrid_gateway.rs
│   └── config/
│       └── database.rs      ← Pool SQLx
└── presentation/
    ├── mod.rs
    ├── http/
    │   ├── router.rs        ← axum Router com rotas
    │   ├── handlers/
    │   │   └── order_handler.rs
    │   └── schemas/         ← Structs Serde para req/resp HTTP
    │       └── order_schema.rs
    └── error_mapping.rs     ← DomainError → StatusCode
```

---

## Entidade de Domínio

```rust
// src/domain/entities/order.rs
use crate::domain::{
    errors::DomainError,
    value_objects::{CustomerId, Money, OrderId, ProductId},
};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct Order {
    id: OrderId,
    customer_id: CustomerId,
    items: Vec<OrderItem>,
    status: OrderStatus,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    Draft,
    Confirmed,
    Cancelled,
}

impl Order {
    pub fn create(customer_id: CustomerId) -> Self {
        Self {
            id: OrderId::generate(),
            customer_id,
            items: Vec::new(),
            status: OrderStatus::Draft,
            created_at: Utc::now(),
        }
    }

    pub fn add_item(
        &mut self,
        product_id: ProductId,
        quantity: u32,
        unit_price: Money,
    ) -> Result<(), DomainError> {
        if self.status != OrderStatus::Draft {
            return Err(DomainError::InvalidStateTransition {
                entity: "Order",
                action: "add_item",
                current_state: format!("{:?}", self.status),
            });
        }
        if quantity == 0 {
            return Err(DomainError::InvalidValue {
                field: "quantity",
                reason: "must be greater than zero".into(),
            });
        }
        self.items.push(OrderItem { product_id, quantity, unit_price });
        Ok(())
    }

    pub fn confirm(&mut self) -> Result<(), DomainError> {
        if self.status != OrderStatus::Draft {
            return Err(DomainError::InvalidStateTransition {
                entity: "Order",
                action: "confirm",
                current_state: format!("{:?}", self.status),
            });
        }
        if self.items.is_empty() {
            return Err(DomainError::BusinessRuleViolation(
                "Cannot confirm an empty order".into(),
            ));
        }
        self.status = OrderStatus::Confirmed;
        Ok(())
    }

    // Getters
    pub fn id(&self) -> &OrderId { &self.id }
    pub fn status(&self) -> &OrderStatus { &self.status }
    pub fn items(&self) -> &[OrderItem] { &self.items }
}
```

---

## Value Object — Newtype Pattern

```rust
// src/domain/value_objects/order_id.rs
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Newtype para evitar confusão entre IDs de tipos diferentes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(Uuid);

impl OrderId {
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_str(s: &str) -> Result<Self, DomainError> {
        Uuid::parse_str(s)
            .map(Self)
            .map_err(|_| DomainError::InvalidValue {
                field: "OrderId",
                reason: format!("invalid UUID format: {s}"),
            })
    }

    pub fn value(&self) -> Uuid { self.0 }
}

impl std::fmt::Display for OrderId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
```

---

## Erros de Domínio (thiserror)

```rust
// src/domain/errors.rs
use thiserror::Error;

/// Hierarquia de erros do domínio — sem dependências externas.
#[derive(Debug, Error)]
pub enum DomainError {
    #[error("{entity} not found: id={id}")]
    NotFound { entity: &'static str, id: String },

    #[error("Business rule violated: {0}")]
    BusinessRuleViolation(String),

    #[error("Invalid state transition: cannot {action} {entity} in state {current_state}")]
    InvalidStateTransition {
        entity: &'static str,
        action: &'static str,
        current_state: String,
    },

    #[error("Invalid value for {field}: {reason}")]
    InvalidValue { field: &'static str, reason: String },

    #[error("Conflict: {entity} with id={id} already exists")]
    AlreadyExists { entity: &'static str, id: String },
}
```

---

## Port Trait (Repository)

```rust
// src/domain/repositories/order_repository.rs
use async_trait::async_trait;
use crate::domain::{entities::Order, errors::DomainError, value_objects::OrderId};

/// Port secundário — trait puro, sem dependência de banco.
#[async_trait]
pub trait OrderRepository: Send + Sync {
    async fn find_by_id(&self, id: &OrderId) -> Result<Option<Order>, DomainError>;
    async fn save(&self, order: &Order) -> Result<(), DomainError>;
    async fn delete(&self, id: &OrderId) -> Result<(), DomainError>;
    async fn exists(&self, id: &OrderId) -> Result<bool, DomainError>;
}
```

---

## Use Case

```rust
// src/application/use_cases/process_order.rs
use std::sync::Arc;
use tracing::{info, instrument};
use crate::domain::{errors::DomainError, repositories::order_repository::OrderRepository, value_objects::OrderId};
use super::super::dtos::process_order::{ProcessOrderRequest, ProcessOrderResponse};

pub struct ProcessOrderUseCase {
    order_repository: Arc<dyn OrderRepository>,
}

impl ProcessOrderUseCase {
    pub fn new(order_repository: Arc<dyn OrderRepository>) -> Self {
        Self { order_repository }
    }

    #[instrument(skip(self), fields(order_id = %request.order_id))]
    pub async fn execute(&self, request: ProcessOrderRequest) -> Result<ProcessOrderResponse, DomainError> {
        info!("Processing order");

        let order_id = OrderId::from_str(&request.order_id)?;

        let mut order = self.order_repository
            .find_by_id(&order_id)
            .await?
            .ok_or_else(|| DomainError::NotFound { entity: "Order", id: request.order_id.clone() })?;

        order.confirm()?;

        self.order_repository.save(&order).await?;

        info!("Order processed successfully");
        Ok(ProcessOrderResponse {
            order_id: order.id().to_string(),
            status: format!("{:?}", order.status()),
        })
    }
}
```

---

## Handler Axum (Apresentação)

```rust
// src/presentation/http/handlers/order_handler.rs
use axum::{extract::{Path, State}, http::StatusCode, Json, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::{application::use_cases::process_order::ProcessOrderUseCase, domain::errors::DomainError};

#[derive(Deserialize)]
pub struct ProcessOrderBody {
    pub payment_method: String,
}

#[derive(Serialize)]
pub struct OrderResponse {
    pub order_id: String,
    pub status: String,
}

pub async fn process_order(
    Path(order_id): Path<String>,
    State(use_case): State<Arc<ProcessOrderUseCase>>,
    Json(body): Json<ProcessOrderBody>,
) -> impl IntoResponse {
    match use_case.execute(ProcessOrderRequest { order_id, payment_method: body.payment_method }).await {
        Ok(resp) => (StatusCode::OK, Json(OrderResponse { order_id: resp.order_id, status: resp.status })).into_response(),
        Err(e) => domain_error_to_response(e),
    }
}

fn domain_error_to_response(err: DomainError) -> axum::response::Response {
    let (status, code) = match &err {
        DomainError::NotFound { .. } => (StatusCode::NOT_FOUND, "NOT_FOUND"),
        DomainError::AlreadyExists { .. } => (StatusCode::CONFLICT, "CONFLICT"),
        DomainError::InvalidValue { .. } => (StatusCode::BAD_REQUEST, "INVALID_VALUE"),
        _ => (StatusCode::UNPROCESSABLE_ENTITY, "BUSINESS_RULE_VIOLATION"),
    };
    (status, Json(serde_json::json!({ "error": err.to_string(), "code": code }))).into_response()
}
```

---

## Testes

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirm_should_succeed_when_items_present() {
        let mut order = Order::create(CustomerId::generate());
        order.add_item(ProductId::generate(), 2, Money::new(100, "BRL")).unwrap();

        let result = order.confirm();

        assert!(result.is_ok());
        assert_eq!(order.status(), &OrderStatus::Confirmed);
    }

    #[test]
    fn confirm_should_fail_when_order_is_empty() {
        let mut order = Order::create(CustomerId::generate());

        let result = order.confirm();

        assert!(matches!(result, Err(DomainError::BusinessRuleViolation(_))));
    }

    // Teste de Use Case com mock (mockall)
    #[tokio::test]
    async fn execute_should_confirm_and_save_order() {
        let mut mock_repo = MockOrderRepository::new();
        let order = Order::create(CustomerId::generate());
        let order_id = order.id().to_string();

        mock_repo.expect_find_by_id()
            .returning(move |_| Ok(Some(order.clone())));
        mock_repo.expect_save()
            .returning(|_| Ok(()));

        let use_case = ProcessOrderUseCase::new(Arc::new(mock_repo));
        let result = use_case.execute(ProcessOrderRequest { order_id: order_id.clone(), payment_method: "card".into() }).await;

        assert!(result.is_ok());
    }
}
```

---

## Checklist de Qualidade

- [ ] `domain/` sem dependências de `axum`, `sqlx`, `tokio` diretas (apenas `async_trait`)
- [ ] Erros de domínio com `thiserror` (não `anyhow` no domínio)
- [ ] Newtype pattern para todos os IDs e Value Objects
- [ ] `Arc<dyn Trait>` para injeção de dependência
- [ ] `#[instrument]` nos use cases para tracing automático
- [ ] `Result<T, DomainError>` em todo o domínio (sem `unwrap/expect` em produção)
- [ ] `clippy::all` + `clippy::pedantic` sem warnings
- [ ] Testes unitários sem `tokio` onde possível (lógica síncrona)
- [ ] `mockall` para mocks de traits assíncronos nos testes de use case
