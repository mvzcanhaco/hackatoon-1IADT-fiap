# Skill: Observabilidade

> Referências: [OpenTelemetry Docs](https://opentelemetry.io/docs/), [Prometheus Best Practices](https://prometheus.io/docs/practices/), [Google SRE Book](https://sre.google/sre-book/table-of-contents/)

---

## Os 3 Pilares da Observabilidade

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│    LOGS     │   │   METRICS   │   │   TRACES    │
│             │   │             │   │             │
│ O que       │   │ O quanto    │   │ Por onde    │
│ aconteceu   │   │ aconteceu   │   │ passou      │
│             │   │             │   │             │
│ Eventos     │   │ Agregações  │   │ Contexto    │
│ estruturados│   │ numéricas   │   │ distribuído │
└─────────────┘   └─────────────┘   └─────────────┘
     Loki               Prometheus        Jaeger
     ELK Stack          Grafana           Zipkin
     CloudWatch         Datadog           OTEL Collector
```

---

## Logs Estruturados

### Princípios

```
✅ Log como JSON (nunca como string concatenada em produção)
✅ Contexto rico: request_id, user_id, trace_id em todo log
✅ Nível correto: DEBUG (dev), INFO (operações normais), WARN (recuperável), ERROR (requer ação)
❌ Nunca logar dados sensíveis: senha, token, CPF, cartão, chave privada
❌ Nunca usar print/console.log em produção — use o logger configurado
```

### Campos Obrigatórios por Log

```json
{
  "timestamp": "2024-05-14T10:30:00.000Z",
  "level": "INFO",
  "service": "order-service",
  "version": "1.2.3",
  "environment": "production",
  "trace_id": "abc123def456",
  "span_id": "789xyz",
  "request_id": "req-uuid-here",
  "message": "Order confirmed",
  "order_id": "order-uuid",
  "customer_id": "cust-uuid",
  "duration_ms": 42
}
```

### Padrão por Linguagem

**Python (structlog + pino)**
```python
import structlog

logger = structlog.get_logger()

# Contexto binding — propaga em toda a request
log = logger.bind(
    request_id=request_id,
    user_id=user_id,
    trace_id=trace_id,
)

log.info("order.confirmed", order_id=str(order.id), total_cents=order.total_cents)
log.error("order.confirm_failed", order_id=str(order_id), error=str(exc))
```

**TypeScript (pino)**
```typescript
const log = logger.child({ requestId, userId, traceId });

log.info({ orderId, totalCents }, "order.confirmed");
log.error({ orderId, err }, "order.confirm_failed");
```

**Java (SLF4J + Logback JSON)**
```java
private static final Logger log = LoggerFactory.getLogger(ProcessOrderUseCase.class);

// MDC injeta contexto automaticamente em todos os logs da thread
MDC.put("requestId", requestId);
MDC.put("userId", userId);

log.info("order.confirmed orderId={} totalCents={}", orderId, totalCents);
log.error("order.confirm_failed orderId={}", orderId, exc);
```

### O que NÃO logar

```python
# ❌ NUNCA — dados sensíveis
log.info("User logged in", password=user.password, token=jwt_token)
log.info("Payment", card_number="4111-1111-1111-1111", cvv="123")

# ✅ SEMPRE mascarar
log.info("User logged in", user_id=user.id, email=mask_email(user.email))
log.info("Payment processed", payment_id=payment.id, last_four=card[-4:])
```

---

## Métricas com Prometheus

### Tipos de Métricas

| Tipo | Quando Usar | Exemplo |
|------|------------|---------|
| **Counter** | Algo que só aumenta | requests totais, erros |
| **Gauge** | Valor que sobe e desce | conexões ativas, uso de memória |
| **Histogram** | Distribuição de valores | latência de requests |
| **Summary** | Percentis pré-calculados | latência p95, p99 |

### Métricas Obrigatórias por Serviço (RED Method)

```
R — Rate: requisições por segundo
E — Errors: taxa de erros (5xx, 4xx de negócio)
D — Duration: latência (histogram p50, p95, p99)
```

### Implementação por Linguagem

**Python (prometheus-client)**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Contador de requests
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

# Histograma de latência
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Gauge de recursos
active_connections = Gauge("active_connections", "Active DB connections")

# Uso
with http_request_duration_seconds.labels(method="POST", endpoint="/orders").time():
    result = use_case.execute(request)
http_requests_total.labels(method="POST", endpoint="/orders", status_code=201).inc()
```

**Go**
```go
var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{Name: "http_requests_total"},
        []string{"method", "endpoint", "status_code"},
    )
    httpDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
)
```

### Queries Prometheus Essenciais

```promql
# Taxa de erros 5xx (últimos 5 min)
rate(http_requests_total{status_code=~"5.."}[5m])
/ rate(http_requests_total[5m])

# Latência p95 por endpoint
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m])
)

# Disponibilidade das últimas 24h
avg_over_time(
  (rate(http_requests_total{status_code!~"5.."}[5m])
  / rate(http_requests_total[5m]))[24h:5m]
)
```

---

## Tracing Distribuído com OpenTelemetry

### Por que usar OTEL?

OpenTelemetry é o padrão aberto vendor-neutral. Instrumente uma vez, envie para Jaeger, Zipkin, Datadog, Grafana Tempo — qualquer backend.

### Propagação de Contexto

```
HTTP Request ──► Service A ──► Service B ──► Service C
                    │               │              │
              [Span A-1]       [Span B-1]    [Span C-1]
                    └───────────────┴──────────────┘
                              trace_id = "abc123"
```

### Instrumentação Python

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Setup (uma vez na inicialização)
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
)
trace.set_tracer_provider(provider)

# Auto-instrumentação (zero código nos handlers)
FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument(engine=engine)

# Span manual para operações de negócio
tracer = trace.get_tracer("order-service")

class ProcessOrderUseCase:
    def execute(self, request: ProcessOrderRequest) -> ProcessOrderResponse:
        with tracer.start_as_current_span("ProcessOrderUseCase.execute") as span:
            span.set_attribute("order.id", request.order_id)
            span.set_attribute("customer.id", request.customer_id)

            order = self._repository.find_by_id(request.order_id)
            if not order:
                span.set_status(trace.StatusCode.ERROR, "Order not found")
                raise OrderNotFoundError(request.order_id)

            order.confirm()
            self._repository.save(order)

            span.set_attribute("order.status", order.status.value)
            return ProcessOrderResponse(order_id=str(order.id))
```

### Configuração docker-compose (OTEL Stack local)

```yaml
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    volumes:
      - ./otel-config.yaml:/etc/otel/config.yaml
    command: ["--config=/etc/otel/config.yaml"]
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "14250:14250"  # gRPC

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
```

---

## Health Checks

### Níveis de Health

```
/health         → Liveness: o processo está vivo?  (reinicia se falhar)
/health/ready   → Readiness: pode receber tráfego? (tira do load balancer se falhar)
/health/startup → Startup: inicialização completa?  (usado no primeiro boot)
```

### Template de Health Check

```python
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

router = APIRouter(tags=["Health"])

class HealthStatus(BaseModel):
    status: str       # "ok" | "degraded" | "down"
    version: str
    checks: dict[str, str]

@router.get("/health")
async def liveness() -> dict:
    """Liveness: o processo está rodando."""
    return {"status": "ok"}

@router.get("/health/ready")
async def readiness(db: Session = Depends(get_db)) -> HealthStatus:
    """Readiness: verifica dependências críticas."""
    checks = {}

    # Banco de dados
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # Cache (se aplicável)
    try:
        redis_client.ping()
        checks["cache"] = "ok"
    except Exception:
        checks["cache"] = "error"

    all_ok = all(v == "ok" for v in checks.values())

    return HealthStatus(
        status="ok" if all_ok else "degraded",
        version=settings.app_version,
        checks=checks,
    )
```

---

## Dashboard Grafana — Painéis Obrigatórios

```
Painel 1: RED Metrics
  - Requests/segundo (rate)
  - Taxa de erros (%)
  - Latência p50 / p95 / p99

Painel 2: Resources
  - CPU usage (%)
  - Memory usage (%)
  - Active DB connections

Painel 3: Business Metrics
  - [métrica de negócio 1] (ex: pedidos/min)
  - [métrica de negócio 2] (ex: pagamentos processados)
  - Taxa de conversão / sucesso

Painel 4: SLO Compliance
  - Disponibilidade (%) nas últimas 24h / 7d / 30d
  - Error budget consumido
```

---

## Checklist de Observabilidade

- [ ] Logs em JSON com campos obrigatórios (timestamp, level, trace_id, request_id)
- [ ] Nenhum dado sensível em logs (auditado com grep/SAST)
- [ ] Métricas RED expostas em `/metrics` (Prometheus format)
- [ ] Health checks em `/health` e `/health/ready`
- [ ] Tracing com OpenTelemetry (auto-instrumentação + spans manuais em use cases)
- [ ] trace_id propagado nas respostas HTTP (header `X-Trace-Id`)
- [ ] Dashboard Grafana com painéis RED + Resources + Business
- [ ] Alertas configurados para availability, latência p95, error rate
- [ ] Runbook linkado em cada alerta

---

## Instruções de Uso

```
@workspace #file:.github/copilot/skills/observability.md
#file:src/main.py
#file:.github/copilot/agents/07-devops.md

Configure observabilidade para este serviço.
Stack escolhida: [Prometheus+Grafana | Datadog | CloudWatch]
```
