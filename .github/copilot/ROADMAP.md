# Copilot Dev Framework — ROADMAP

> Versão atual: **v2.2**
> Última atualização: 2025-08

---

## Versões Lançadas

### v1.0 — Base
- Instruções base (`copilot-instructions.md`)
- Agentes: Project Discovery, Architect, Domain Modeler, Developer, Code Reviewer, Test Engineer
- Templates básicos: entity, value-object, use-case, repository
- Perfis: Python

### v2.0 — Anti-Alucinação + Hexagonal
- Protocolo GROUND-TRUTH anti-alucinação
- Template http-controller
- Perfis: Go, Android/Kotlin, C, C++

### v2.1 — TypeScript + Feedback Loops
- Perfil TypeScript/Node.js completo
- Agent 04: distinção DTO vs Schema
- Agent 05: loop de feedback 1ª/2ª revisão com PR templates
- Agent 06: E2E testing com TestClient/httpx
- Agent 07: DevOps Engineer (Dockerfile, CI/CD, health checks)
- Templates: domain-event, port-interface, exceptions

### v2.2 — Skills + SDLC + Linguagens
- **4 Skills**: observability, api-design, event-driven, caching
- **SDLC completo** (6 fases): inception → design → dev → quality gate → release → operate
- **Agent 08**: Data Architect (ERD, migrations, outbox, N+1)
- **Agent 09**: Security Engineer (STRIDE, OWASP Top 10, LGPD)
- **3 Templates**: aggregate-root, query-handler, event-publisher
- **3 Prompts**: security-threat-model, performance-audit, api-contract-design
- **Perfis**: Java 21, Rust 2021, PHP 8.3/Laravel 11

---

## Roadmap Futuro

### v2.3 — GraphQL + gRPC
**Estimativa**: próximo trimestre

- [ ] Skill: GraphQL Design (schema-first, resolvers, DataLoader, N+1 em GraphQL)
- [ ] Skill: gRPC + Protobuf (service definition, streaming, interceptors)
- [ ] Template: GraphQL resolver (hexagonal adapter)
- [ ] Perfil: Go atualizado com gRPC server/client
- [ ] Prompt: `api-contract-graphql.md` — schema design GraphQL-first

### v2.4 — Testing Avançado
**Estimativa**: próximo trimestre

- [ ] Skill: Contract Testing (Pact, consumer-driven contracts)
- [ ] Skill: Mutation Testing (mutmut para Python, PITest para Java)
- [ ] Template: `test-doubles.md` — spy, stub, fake, mock por caso de uso
- [ ] Prompt: `test-strategy.md` — definição da pirâmide por tipo de projeto
- [ ] Agent 06 atualizado: Property-Based Testing (Hypothesis, proptest)

### v2.5 — Plataforma e Infra
**Estimativa**: futuro

- [ ] Skill: Kubernetes (manifests, HPA, PodDisruptionBudget, resource limits)
- [ ] Skill: Terraform (módulos reutilizáveis, state management, drift detection)
- [ ] Template: `service-mesh.md` — Istio/Linkerd circuit breaker, retry, mTLS
- [ ] Agent 07 atualizado: GitOps com Argo CD / Flux
- [ ] Prompt: `infra-as-code-audit.md` — revisão de Terraform e Kubernetes manifests

### v2.6 — DDD Avançado
**Estimativa**: futuro

- [ ] Skill: Event Sourcing (append-only event store, projections, snapshots)
- [ ] Skill: CQRS Avançado (read models com Elasticsearch, materialized views)
- [ ] Template: `saga-orchestrator.md` — saga coordinator com compensações
- [ ] Template: `process-manager.md` — long-running processes
- [ ] Prompt: `bounded-context-map.md` — mapeamento formal de BCs com padrões

### v3.0 — Multi-Agent Collaboration
**Estimativa**: longo prazo

- [ ] Protocolo de handoff entre agentes (Architect → Developer → Reviewer com contexto preservado)
- [ ] Agent Memory: contexto persistido entre sessões (ADRs, decisões passadas)
- [ ] Agent Orchestrator: coordenador que decide qual agente invocar por tipo de tarefa
- [ ] Integração com GitHub Actions: comentários automáticos de review em PRs
- [ ] Métricas de qualidade do framework: hit rate de sugestões aceitas

---

## Contribuição

Para sugerir novas features ou reportar problemas com o framework:

1. Abra uma issue no repositório com label `copilot-framework`
2. Descreva: contexto de uso, problema encontrado, melhoria sugerida
3. Se tiver implementação: abra PR com os arquivos em `.github/copilot/`

### Convenções de Arquivo

```
Skills:      .github/copilot/skills/[nome-kebab-case].md
Agents:      .github/copilot/agents/[nn-nome-kebab-case].md
Templates:   .github/copilot/templates/hexagonal/[camada]/[nome].md
Prompts:     .github/copilot/prompts/[new-project|existing-project]/[nn-nome].md
Linguagens:  .github/copilot/languages/[linguagem]/profile.md
```

### Critérios de Qualidade para Novos Arquivos

- [ ] Referências verificáveis (links para specs, livros, documentações oficiais)
- [ ] Exemplos de código compiláveis e testáveis
- [ ] Checklist ao final do documento
- [ ] Seção "Instruções de Uso" com prompt pronto para copiar
- [ ] Seção "Quando usar/não usar" quando aplicável

---

## Métricas de Adoção

| Métrica | Alvo v2.x | Como Medir |
|---------|-----------|-----------|
| Agentes usados por sprint | ≥ 3 | Survey trimestral |
| Skills referenciadas em PRs | ≥ 1 por feature PR | Labels no GitHub |
| Templates seguidos (% de código) | ≥ 70% | Code review checklist |
| Redução de findings no review | -30% vs baseline | Métricas do Agent 05 |
| TTM (time-to-merge) | -20% | GitHub Insights |
