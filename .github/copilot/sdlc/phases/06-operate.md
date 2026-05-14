# Fase 6: Operate

## Objetivo

Manter o sistema saudável em produção: monitoramento contínuo, resposta a incidentes e ciclo de melhoria contínua (feedback → próxima feature).

**Duração**: Contínua (não tem fim)  
**Agente**: `07-devops`  
**Participantes**: SRE / DevOps / Time de plantão (on-call)

---

## Como Ativar

```
@workspace #file:.github/copilot/sdlc/phases/06-operate.md
           #file:.github/copilot/agents/07-devops.md

Configure observabilidade para [nome do serviço].
Stack: [Prometheus/Grafana | Datadog | CloudWatch | etc.]
```

---

## SLO / SLA — Definição

**SLI** (Service Level Indicator): métrica medida  
**SLO** (Service Level Objective): meta interna  
**SLA** (Service Level Agreement): compromisso com o cliente  

### Template de SLOs por Criticidade

```yaml
# slo.yaml — Definição formal de SLOs
service: [nome-do-servico]
version: "1.0"

slos:
  availability:
    description: "Porcentagem de requests bem-sucedidos (não 5xx)"
    sli: "1 - (rate(http_requests_total{status=~'5..'}[5m]) / rate(http_requests_total[5m]))"
    target: 99.9%        # três noves = max 43min de downtime/mês
    window: 30d

  latency_p95:
    description: "95% dos requests respondem em menos de X ms"
    sli: "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
    target: 500ms
    window: 30d

  latency_p99:
    description: "99% dos requests respondem em menos de X ms"
    sli: "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))"
    target: 2s
    window: 30d

  error_rate:
    description: "Taxa de erros de negócio (4xx excluindo 401/403)"
    sli: "rate(http_requests_total{status=~'4[^01].'} [5m])"
    target: "< 1%"
    window: 7d
```

---

## Alert Rules Obrigatórias

```yaml
# alerting-rules.yml (Prometheus AlertManager)
groups:
  - name: [service]-critical
    rules:
      # Disponibilidade abaixo do SLO
      - alert: HighErrorRate
        expr: |
          (rate(http_requests_total{status=~"5.."}[5m])
          / rate(http_requests_total[5m])) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Taxa de erros 5xx acima de 1% por 2 minutos"
          runbook: "https://wiki/runbooks/high-error-rate"

      # Latência acima do SLO
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Latência p95 acima de 500ms"

      # Serviço indisponível
      - alert: ServiceDown
        expr: up{job="[service]"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "[service] está DOWN"
          runbook: "https://wiki/runbooks/service-down"

      # Uso alto de memória
      - alert: HighMemoryUsage
        expr: |
          container_memory_usage_bytes{name="[service]"}
          / container_spec_memory_limit_bytes > 0.85
        for: 5m
        labels:
          severity: warning
```

---

## Runbook Template

```markdown
# Runbook: [Nome do Alerta]

**Severidade**: 🔴 Critical / 🟠 High / 🟡 Medium
**Alerta**: `[AlertName]`
**Responsável on-call**: ver PagerDuty / rotação

## Sintomas
- [Como o usuário percebe este problema]
- [O que aparece no dashboard]

## Diagnóstico

### Passo 1 — Verificar saúde do serviço
```bash
kubectl get pods -n [namespace] | grep [service]
kubectl describe pod [pod-name] -n [namespace]
```

### Passo 2 — Verificar logs recentes
```bash
kubectl logs -n [namespace] -l app=[service] --since=15m | grep -E "ERROR|CRITICAL"
```

### Passo 3 — Verificar banco de dados
```bash
# Conexões ativas
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
# Queries lentas
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle' ORDER BY duration DESC LIMIT 10;
```

## Causas Conhecidas

| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| OOMKilled | Memory leak ou carga inesperada | Reiniciar pod + investigar |
| Connection refused | Banco indisponível | Verificar status do banco |
| 504 Gateway Timeout | Dependency externa lenta | Verificar timeout + circuit breaker |

## Ação de Mitigação Imediata
[Ação que pode ser feita sem entender a causa raiz — para restaurar o serviço]

## Escalação
- Não resolveu em 15 min → escalar para [pessoa/squad]
- Impacto > X usuários → escalar para [líder técnico]

## Pós-resolução
- Criar post-mortem se downtime > 5 min
- Abrir issue para causa raiz
```

---

## Post-Mortem Template (Blameless)

```markdown
# Post-Mortem: [Título do Incidente]

**Data**: [data]
**Duração do impacto**: [início → fim] = [X minutos]
**Severidade**: [P0 / P1 / P2]
**Usuários afetados**: [estimativa]
**Autor**: [nome] | **Revisado por**: [nome]

## Resumo Executivo
[2-3 frases: o que falhou, por quanto tempo, qual foi o impacto]

## Timeline (UTC)

| Hora | Evento |
|------|--------|
| 14:00 | Deploy de v1.2.3 concluído |
| 14:15 | Alerta HighErrorRate disparado |
| 14:17 | On-call notificado e iniciou investigação |
| 14:25 | Causa raiz identificada: [causa] |
| 14:30 | Rollback para v1.2.2 iniciado |
| 14:33 | Serviço restaurado, erros voltaram ao normal |

## Causa Raiz
[Descrição técnica detalhada da causa — sem culpar pessoas]

## Fator Contribuinte
[O que facilitou a ocorrência do problema: processo, falta de teste, etc.]

## Impacto
- [X] requests falharam
- [Y] usuários afetados
- [Z] transações perdidas / recuperáveis

## O que foi bem
- [Aspecto positivo da resposta ao incidente]
- [Detecção foi rápida graças ao alerta X]

## O que pode melhorar
- [Item identificado, sem julgamento de culpa]

## Ações de Melhoria

| Ação | Responsável | Prazo | Issue |
|------|------------|-------|-------|
| Adicionar teste de integração para [cenário] | [nome] | [data] | #[número] |
| Implementar circuit breaker para [dependency] | [nome] | [data] | #[número] |
| Melhorar runbook para [alerta] | [nome] | [data] | #[número] |
```

---

## Ciclo de Feedback → Próxima Feature

```
Operate → Feedback
    │
    ├── Métricas: qual endpoint tem maior latência? → otimizar
    ├── Erros: quais exceções mais frequentes? → corrigir
    ├── Logs: quais operações falham mais? → investigar
    └── Usuários: feedback via produto → próximo Discovery
          │
          ▼
    Volta para Fase 1 ou Fase 3 (dependendo do escopo)
```

**Reunião semanal de operações** (30min):
- Review dos SLOs da semana
- Top 3 erros mais frequentes
- Alertas disparados e suas causas
- Items para o backlog de confiabilidade

---

## Checklist de Operação Saudável

- [ ] SLOs definidos e monitorados no dashboard
- [ ] Alertas configurados para todas as métricas críticas
- [ ] Runbooks escritos para os alertas mais comuns
- [ ] Rotação de on-call definida e documentada
- [ ] Post-mortems escritos para todos os incidentes P0/P1
- [ ] Ações de melhoria dos post-mortems no backlog
- [ ] Drill de rollback praticado (simulação mensal)
- [ ] Dependências externas com circuit breaker configurado

---

## Próxima Iteração

→ [Fase 1: Inception](01-inception.md) com feedback de produção para próxima feature.  
→ [Fase 3: Development](03-development.md) para hotfixes urgentes.
