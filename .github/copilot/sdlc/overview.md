# SDLC — Diagrama Detalhado

## Fluxo Completo com Agentes e Gates

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COPILOT DEV FRAMEWORK — SDLC v2.2                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  FASE 1: INCEPTION                                              🔍 Discover  │
│                                                                             │
│  Agente: 01-project-discovery.md                                            │
│                                                                             │
│  INPUT: Ideia / problema de negócio                                         │
│                                                                             │
│  ATIVIDADES:                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │   Vision &  │→ │   Domain &  │→ │    NFRs &   │→ │  Priorização &  │   │
│  │  Problema   │  │  Processos  │  │  Restrições │  │   MVP Scope     │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │
│                                                                             │
│  OUTPUT: Discovery Report ─────────────────────────────────────────────┐   │
│                                                                         │   │
│  ⛔ GATE: Definition of Ready (DoR)                                      │   │
│     □ Requisitos em linguagem de negócio                                │   │
│     □ Critérios de aceitação definidos                                  │   │
│     □ MVP scope aprovado                                                │   │
└─────────────────────────────────────────────────────────────────────────┘   │
                                      │ Discovery Report aprovado            │
                                      ▼                                      │
┌─────────────────────────────────────────────────────────────────────────┐  │
│  FASE 2: DESIGN                                                 📐 Design │  │
│                                                                           │  │
│  Agentes: 02-architect, 03-domain-modeler, 08-data-architect             │  │
│                                                                           │  │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────────────────┐   │  │
│  │ Event Storm  │→  │ Bounded Ctxs  │→  │  Hexagonal Architecture  │   │  │
│  │ (Domínio)   │   │ + Aggregates  │   │  + Ports & Adapters      │   │  │
│  └──────────────┘   └───────────────┘   └───────────────────────────┘   │  │
│          │                                           │                   │  │
│          ▼                                           ▼                   │  │
│  ┌──────────────┐                         ┌───────────────────────────┐  │  │
│  │ Domain Model │                         │  Database Schema + ERD    │  │  │
│  │ (Entities,   │                         │  + Migrations Strategy    │  │  │
│  │  VOs, Events)│                         │  (Agent 08)               │  │  │
│  └──────────────┘                         └───────────────────────────┘  │  │
│                                                                           │  │
│  OUTPUT: Architecture Document + Domain Model + Schema Design             │  │
│                                                                           │  │
│  ⛔ GATE: Architecture Review                                              │  │
│     □ ADRs documentados para decisões chave                              │  │
│     □ Nenhum framework no domínio                                        │  │
│     □ Schema backward compatible                                         │  │
└───────────────────────────────────────────────────────────────────────────┘  │
                                      │ Architecture aprovada                 │
                                      ▼                                      │
┌─────────────────────────────────────────────────────────────────────────┐  │
│  FASE 3: DEVELOPMENT                                            💻 Build  │  │
│                                                                           │  │
│  Agentes: 04-developer, 09-security-engineer                             │  │
│                                                                           │  │
│  Inner Loop (repete por story/feature):                                  │  │
│                                                                           │  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────────┐  │  │
│  │ Implement│→  │  Lint +  │→  │  Tests   │→  │   Security Review    │  │  │
│  │ (inside- │   │  Format  │   │  (unit + │   │   (Agent 09) para    │  │  │
│  │  out)    │   │          │   │  integ.) │   │   código crítico     │  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────────┘  │  │
│       ▲                                                    │             │  │
│       └────────────── fix & iterate ───────────────────────┘             │  │
│                                                                           │  │
│  Ordem de implementação (inside-out):                                    │  │
│  1. Domain (entities, VOs, exceptions)                                   │  │
│  2. Application (use cases + DTOs)                                       │  │
│  3. Infrastructure (repositories, gateways)                              │  │
│  4. Presentation (controllers, schemas)                                  │  │
│                                                                           │  │
│  OUTPUT: PR aberto com feature completa                                   │  │
└───────────────────────────────────────────────────────────────────────────┘  │
                                      │ PR aberto                             │
                                      ▼                                      │
┌─────────────────────────────────────────────────────────────────────────┐  │
│  FASE 4: QUALITY GATE                                           ✅ Verify  │  │
│                                                                           │  │
│  Agentes: 05-code-reviewer, 06-test-engineer, 09-security-engineer      │  │
│                                                                           │  │
│  ┌────────────────────┐  ┌──────────────────┐  ┌────────────────────┐   │  │
│  │  Code Review       │  │  Test Coverage   │  │  Security Scan     │   │  │
│  │  (5 camadas)       │  │  ≥ 80% no código │  │  OWASP checklist   │   │  │
│  │  + feedback loop   │  │  novo            │  │  + LGPD audit      │   │  │
│  └────────────────────┘  └──────────────────┘  └────────────────────┘   │  │
│                                                                           │  │
│  ⛔ GATE: Quality Gate — TODOS os critérios devem passar:                │  │
│     □ Zero findings críticos ou altos no code review                    │  │
│     □ Cobertura ≥ 80% no código novo                                     │  │
│     □ Zero vulnerabilidades críticas (OWASP)                            │  │
│     □ Testes E2E dos fluxos críticos passando                           │  │
│     □ DoD verificado item a item                                         │  │
│                                                                           │  │
│  Resultado: PR aprovado → merge para main                                │  │
└───────────────────────────────────────────────────────────────────────────┘  │
                                      │ Merge aprovado                        │
                                      ▼                                      │
┌─────────────────────────────────────────────────────────────────────────┐  │
│  FASE 5: RELEASE                                                🚀 Ship   │  │
│                                                                           │  │
│  Agente: 07-devops                                                       │  │
│                                                                           │  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  CI Pipeline │→ │  Build Image │→ │  Deploy to   │→ │ Smoke Tests │  │  │
│  │  (lint+test+ │  │  + tag + push│  │  Staging /   │  │ (health +   │  │  │
│  │  scan verde) │  │  to registry │  │  Production  │  │  key flows) │  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │  │
│                                                                           │  │
│  ⚠️ Smoke test falhou?                                                    │  │
│  ┌────────────────────────────────────────────────────────────────────┐  │  │
│  │  ROLLBACK AUTOMÁTICO → versão anterior restaurada em < 5 min      │  │  │
│  └────────────────────────────────────────────────────────────────────┘  │  │
│                                                                           │  │
│  OUTPUT: Release Notes + Changelog + Deploy confirmado                   │  │
└───────────────────────────────────────────────────────────────────────────┘  │
                                      │ Deploy confirmado                     │
                                      ▼                                      │
┌─────────────────────────────────────────────────────────────────────────┐  │
│  FASE 6: OPERATE                                               📊 Monitor  │  │
│                                                                           │  │
│  Agente: 07-devops                                                       │  │
│                                                                           │  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │  SLO Monitor │  │  Alert Rules │  │   On-Call    │  │  Post-mortem│  │  │
│  │  (error rate │  │  (latência,  │  │   Runbook    │  │  (blameless)│  │  │
│  │   latência)  │  │   erros,     │  │              │  │             │  │  │
│  │              │  │   recursos)  │  │              │  │             │  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │  │
│                                                                           │  │
│  ──────────────────────────────────────────────────────────────────────  │  │
│                                                                           │  │
│  FEEDBACK → próxima feature volta para ────────────────────────────────► │  │
│             FASE 3 (com Discovery já feito)                              │  │
└───────────────────────────────────────────────────────────────────────────┘  │
                                                                               │
◄──────────────────────────────── CICLO CONTÍNUO ──────────────────────────────┘
```

---

## Mapa de Agentes por Fase

```
                    AGENTES DO FRAMEWORK
┌──────┬────────────────────────────────────────────────────────────────┐
│ Fase │ Agentes Ativos                                                  │
├──────┼────────────────────────────────────────────────────────────────┤
│  1   │ 01-project-discovery                                           │
│  2   │ 02-architect + 03-domain-modeler + 08-data-architect          │
│  3   │ 04-developer + 09-security-engineer                           │
│  4   │ 05-code-reviewer + 06-test-engineer + 09-security-engineer    │
│  5   │ 07-devops                                                      │
│  6   │ 07-devops                                                      │
└──────┴────────────────────────────────────────────────────────────────┘
```

---

## Métricas do SDLC

| Métrica | Meta | Alerta |
|---------|------|--------|
| Tempo médio por fase 1-2 | ≤ 3 dias | > 5 dias |
| Tempo médio de code review | ≤ 4h | > 24h |
| Taxa de PRs com finding crítico | < 5% | > 15% |
| Cobertura média | ≥ 80% | < 70% |
| Tempo de deploy (CI → prod) | ≤ 30 min | > 1h |
| MTTR (tempo de recovery) | ≤ 30 min | > 2h |
| Taxa de rollback | < 2% | > 5% |
