# Skill: Caching

> Referências: [Redis Documentation](https://redis.io/docs/), [Martin Fowler — Patterns of Enterprise Application Architecture](https://martinfowler.com/eaaCatalog/), [AWS Caching Best Practices](https://aws.amazon.com/caching/)

---

## Quando Usar Cache

```
✅ Dados lidos com alta frequência e baixa taxa de mudança
✅ Computações custosas que podem ser reutilizadas
✅ Resultados de chamadas externas (APIs de terceiros)
✅ Sessões de usuário
✅ Rate limiting (contadores)
✅ Dados de configuração e feature flags

❌ Dados que mudam frequentemente (invalidação custosa supera o benefício)
❌ Dados únicos por usuário em grande volume (cache pollution)
❌ Dados sensíveis sem criptografia (tokens, senhas)
❌ Operações de escrita (nunca escreva no banco via cache)
```

---

## Padrões de Cache

### Cache-Aside (Lazy Loading) — O mais comum

```
Leitura:
  1. App verifica cache → HIT? retorna valor
  2. MISS → App lê do banco
  3. App popula o cache
  4. App retorna valor ao cliente

Escrita:
  1. App escreve no banco
  2. App invalida/atualiza o cache (não escreve no cache diretamente)
```

```python
class OrderRepository:
    def find_by_id(self, order_id: OrderId) -> Optional[Order]:
        cache_key = f"order:{order_id}"

        # 1. Tentar cache
        cached = self._redis.get(cache_key)
        if cached:
            return Order.from_dict(json.loads(cached))  # HIT

        # 2. MISS — buscar no banco
        order = self._db_repo.find_by_id(order_id)

        # 3. Popular cache (com TTL)
        if order:
            self._redis.setex(
                cache_key,
                ex=300,  # 5 minutos
                value=json.dumps(order.to_dict()),
            )

        return order  # pode ser None

    def save(self, order: Order) -> None:
        self._db_repo.save(order)
        # 4. Invalidar cache após escrita
        self._redis.delete(f"order:{order.id}")
```

### Write-Through — Escrita simultânea em cache e banco

```
Escrita:
  1. App escreve no cache
  2. Cache escreve no banco (sincronamente)
  3. Retorna ao cliente

Vantagem: cache sempre consistente
Desvantagem: latência de escrita aumenta
```

### Write-Behind (Write-Back) — Banco atualizado assincronamente

```
Escrita:
  1. App escreve no cache
  2. Cache confirma ao cliente imediatamente
  3. Cache escreve no banco assincronamente (em batch)

Vantagem: baixíssima latência de escrita
Desvantagem: risco de perda de dados se Redis cair antes de persistir
Uso: métricas, contadores, dados não-críticos
```

### Read-Through — Cache como proxy transparente

```
App → Cache → Banco
              (cache busca automaticamente no MISS)

Implementado por: Redis com módulo de read-through (ex: RedisGears)
Mais simples para o app: não precisa gerenciar MISS manualmente
```

---

## TTL — Estratégias de Expiração

```python
# TTL por tipo de dado
CACHE_TTL = {
    "session": 3600,        # 1 hora — sessão de usuário
    "user_profile": 300,    # 5 min — perfil pode mudar
    "order": 60,            # 1 min — pedido em processamento
    "product_catalog": 3600,  # 1h — raramente muda
    "config": 86400,        # 24h — configurações do sistema
    "rate_limit": 60,       # 1 min — janela de rate limiting
}

# Jitter para evitar thundering herd (todos expirando ao mesmo tempo)
import random

def ttl_with_jitter(base_ttl: int, jitter_pct: float = 0.1) -> int:
    jitter = int(base_ttl * jitter_pct)
    return base_ttl + random.randint(-jitter, jitter)

redis.setex(key, ex=ttl_with_jitter(300), value=data)
```

---

## Cache de Queries Complexas

```python
# Cache de resultado de query cara
class ReportRepository:
    def get_monthly_summary(self, year: int, month: int) -> MonthlySummary:
        cache_key = f"report:monthly:{year}:{month}"

        cached = self._redis.get(cache_key)
        if cached:
            return MonthlySummary.from_dict(json.loads(cached))

        # Query custosa no banco
        result = self._db.execute("""
            SELECT
                count(*) as order_count,
                sum(total_cents) as revenue_cents
            FROM orders
            WHERE status = 'confirmed'
              AND date_trunc('month', created_at) = make_date(%s, %s, 1)
        """, (year, month))

        summary = MonthlySummary(
            order_count=result.order_count,
            revenue_cents=result.revenue_cents,
        )

        # TTL longo para mês fechado, curto para mês atual
        from datetime import date
        is_current_month = (date.today().year, date.today().month) == (year, month)
        ttl = 60 if is_current_month else 86400  # 1min vs 24h

        self._redis.setex(cache_key, ex=ttl, value=json.dumps(summary.to_dict()))
        return summary
```

---

## Rate Limiting com Redis

```python
import time

def is_rate_limited(
    redis_client,
    key: str,
    limit: int,
    window_seconds: int,
) -> bool:
    """
    Sliding window rate limiting usando Redis sorted set.
    Retorna True se o request deve ser bloqueado.
    """
    now = time.time()
    window_start = now - window_seconds

    pipe = redis_client.pipeline()
    # Remove entradas fora da janela
    pipe.zremrangebyscore(key, 0, window_start)
    # Adiciona o request atual
    pipe.zadd(key, {str(now): now})
    # Conta requests na janela
    pipe.zcard(key)
    # Expira a chave automaticamente
    pipe.expire(key, window_seconds)
    results = pipe.execute()

    request_count = results[2]
    return request_count > limit

# Uso no middleware
def rate_limit_middleware(request, limit=100, window=60):
    key = f"rate_limit:{request.client.host}"
    if is_rate_limited(redis, key, limit, window):
        raise HTTPException(429, "Too Many Requests",
                           headers={"Retry-After": str(window)})
```

---

## Cache de Sessão

```python
import secrets
from datetime import timedelta

class SessionStore:
    SESSION_PREFIX = "session:"
    SESSION_TTL = int(timedelta(hours=2).total_seconds())

    def create(self, user_id: str, data: dict) -> str:
        session_id = secrets.token_urlsafe(32)
        key = f"{self.SESSION_PREFIX}{session_id}"

        self._redis.setex(
            key,
            ex=self.SESSION_TTL,
            value=json.dumps({"user_id": user_id, **data}),
        )
        return session_id

    def get(self, session_id: str) -> Optional[dict]:
        key = f"{self.SESSION_PREFIX}{session_id}"
        data = self._redis.get(key)
        if data:
            # Renovar TTL a cada acesso (sliding expiration)
            self._redis.expire(key, self.SESSION_TTL)
            return json.loads(data)
        return None

    def invalidate(self, session_id: str) -> None:
        self._redis.delete(f"{self.SESSION_PREFIX}{session_id}")
```

---

## Cache Warming (Pré-aquecimento)

Evita thundering herd após deploy ou reinício:

```python
async def warm_cache() -> None:
    """
    Popula cache com dados frequentes antes de abrir tráfego.
    Chamado durante startup, antes do health check ficar ready.
    """
    logger.info("Warming cache...")

    # Carregar configurações
    configs = await config_repo.find_all()
    for config in configs:
        redis.setex(f"config:{config.key}", ex=86400, value=json.dumps(config.value))

    # Carregar top produtos
    top_products = await product_repo.find_top(limit=100)
    for product in top_products:
        redis.setex(f"product:{product.id}", ex=3600, value=json.dumps(product.to_dict()))

    logger.info("Cache warmed", config_count=len(configs), product_count=len(top_products))
```

---

## Anti-Padrões de Cache

```
❌ Cache de dados sem TTL (nunca expira → dados stale para sempre)
❌ Cache key sem namespace (colisões entre serviços)
❌ Cachear exceções ou None sem cuidado (negative caching deveria ser explícito)
❌ Cache como fonte de verdade (banco é sempre a fonte canônica)
❌ Distribuir lógica de invalidação por todo o código (centralizar em repositórios)
❌ Cachear objetos com referências a conexões ou resources externos
❌ Cache key com dados do usuário sem isolamento (vazar dados entre usuários)
```

---

## Configuração Redis (Python)

```python
import redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

# Cliente com retry automático
redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=settings.redis_password,
    db=0,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
    retry=Retry(ExponentialBackoff(), retries=3),
    retry_on_error=[ConnectionError, TimeoutError],
)

# Health check
def check_redis() -> bool:
    try:
        return redis_client.ping()
    except Exception:
        return False
```

---

## Checklist de Cache

- [ ] Cache-key com namespace por serviço (evita colisões)
- [ ] TTL definido para todas as chaves (nenhuma chave sem expiração)
- [ ] Jitter no TTL para evitar thundering herd
- [ ] Invalidação centralizada nos repositórios (não espalhada no código)
- [ ] Dados sensíveis nunca em texto plano no cache (criptografar se necessário)
- [ ] Retry com backoff configurado para falhas de Redis
- [ ] Health check inclui verificação do Redis
- [ ] Métricas: cache hit rate, miss rate, latência
- [ ] Cache warming no startup para dados frequentes
- [ ] Teste: comportamento correto com Redis indisponível (degradação graciosa)

---

## Instruções de Uso

```
@workspace #file:.github/copilot/skills/caching.md
#file:src/infrastructure/repositories/[repo].py
#file:.github/copilot/agents/08-data-architect.md

Implemente cache para [recurso] no repositório [RepoName].
Padrão: [cache-aside | write-through]
TTL: [definir por tipo de dado]
```
