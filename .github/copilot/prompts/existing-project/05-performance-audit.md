# Prompt: Performance Audit (Projeto Existente)

> Use este prompt para auditar performance de um serviço em produção ou em pré-produção.
> Referência: [.github/copilot/agents/08-data-architect.md], [.github/copilot/skills/caching.md], [.github/copilot/skills/observability.md]

---

## Contexto Necessário

```
@workspace #file:.github/copilot/prompts/existing-project/05-performance-audit.md
           #file:.github/copilot/agents/08-data-architect.md
           #file:.github/copilot/skills/caching.md
           #file:.github/copilot/skills/observability.md
           #file:src/infrastructure/repositories/[repositório_mais_crítico].py
           #file:src/application/use_cases/[use_case_mais_usado].py
```

---

## Prompt

```
Aja como um engenheiro de performance sênior e execute uma auditoria completa do serviço descrito.

## Serviço

Nome: [NomeDoServiço]
Linguagem/Stack: [Python/FastAPI | Node.js/Express | Java/Spring | ...]
Banco de dados: [PostgreSQL | MySQL | MongoDB | ...]
Cache: [Redis | Memcached | Sem cache]
Volume atual: [requisições/minuto, usuários ativos]
Problemas reportados: [lentidão em X, timeout em Y, alto uso de memória]

## Métricas Atuais (se disponíveis)

```
p50 latência: [Xms]
p95 latência: [Xms]
p99 latência: [Xms]
Error rate: [X%]
Throughput: [req/s]
```

## Código para Análise

[Cole aqui os arquivos mais relevantes ou inclua via #file:]

---

## Entregáveis da Auditoria

### 1. Hotspots de Performance

Analise o código fornecido e identifique:

**Problemas de banco de dados:**
  - [ ] N+1 queries (loop com SELECT dentro)
  - [ ] Queries sem índice nos campos de filtro/ORDER BY
  - [ ] SELECT * onde apenas alguns campos são usados
  - [ ] Ausência de paginação em listagens (full table scan)
  - [ ] Transações muito longas segurando locks
  - [ ] Conexões não retornadas ao pool

**Problemas de aplicação:**
  - [ ] Computações custosas sem cache (regex complexo, criptografia em loop)
  - [ ] I/O síncrono onde poderia ser assíncrono
  - [ ] Objetos grandes sendo copiados em memória desnecessariamente
  - [ ] Serialização/deserialização excessiva (to_dict/from_dict em loop)
  - [ ] Logging excessivo em paths críticos

**Problemas de cache:**
  - [ ] Cache miss rate alto (TTL muito curto ou chave incorreta)
  - [ ] Cache stampede (thundering herd) sem jitter
  - [ ] Dados não cacheáveis sendo cacheados
  - [ ] Dados que deveriam ser cacheados mas não são

**Problemas de arquitetura:**
  - [ ] Chamadas síncronas para serviços externos no critical path
  - [ ] Fanout excessivo (um request dispara muitas chamadas downstream)
  - [ ] Payload muito grande sendo transferido desnecessariamente

### 2. Análise de Queries SQL

Para cada query identificada como potencialmente lenta:

```sql
-- Query original
[SQL aqui]

-- Índice sugerido
CREATE INDEX CONCURRENTLY idx_[tabela]_[campos] ON [tabela]([campos])
WHERE [condição_parcial_se_aplicável];

-- Query otimizada (se possível)
[SQL otimizado aqui]

-- Estimativa de melhoria: [X]x mais rápida
```

Use `EXPLAIN ANALYZE` para confirmar antes de criar índices em produção.

### 3. Oportunidades de Cache

Para cada endpoint/operação identificada:

| Operação | Frequência | TTL Sugerido | Padrão | Invalidação |
|---------|-----------|--------------|--------|-------------|
| [op 1] | [req/min] | [Xs] | Cache-Aside | [evento que invalida] |

### 4. Plano de Otimização Priorizado

Ordene por: impacto × facilidade de implementação

**🔴 Alta prioridade (Quick Wins):**
  1. [mudança com maior impacto e menor risco]
  2. [...]

**🟠 Média prioridade:**
  1. [mudança que requer teste mais cuidadoso]
  2. [...]

**🟡 Baixa prioridade (longo prazo):**
  1. [refatoração arquitetural ou infra]
  2. [...]

### 5. Métricas de Baseline e Target

| Métrica | Atual | Target | Como Medir |
|---------|-------|--------|-----------|
| p95 latência endpoint X | [Xms] | [Yms] | Prometheus histogram |
| Cache hit rate | [X%] | [Y%] | Redis INFO stats |
| N+1 queries eliminadas | [N] | 0 | Slow query log |
| Throughput máximo | [req/s] | [req/s] | Load test (k6/locust) |

### 6. Código Otimizado

Para os 2-3 hotspots mais críticos, forneça:

```python
# ANTES (problemático)
[código original com o problema]

# DEPOIS (otimizado)
[código corrigido com explicação da melhoria]
```

### 7. Load Test Script

```python
# tests/load/test_[endpoint].py usando locust ou k6
[script de load test para validar as melhorias]
```

---

## Checklist de Pré-Produção

Antes de colocar as otimizações em produção:
  - [ ] Benchmark executado em ambiente isolado (não produção)
  - [ ] Índices criados com CONCURRENTLY (sem lock na tabela)
  - [ ] Feature flag para rollback das mudanças de cache
  - [ ] Monitoramento ativado antes do deploy (métricas de baseline capturadas)
  - [ ] Rollback plan documentado

---

## Quando Usar Este Prompt

```
✅ p95 > 500ms em endpoints críticos
✅ Timeout errors > 0.1% das requisições
✅ CPU/Memória consistentemente > 70%
✅ Antes de campanhas ou eventos de alto volume
✅ Após crescimento 5x+ no volume de dados
```
