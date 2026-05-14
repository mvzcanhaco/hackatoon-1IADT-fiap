# SDLC — Software Development Life Cycle

Framework estruturado para conduzir projetos de software do zero ao ar, com gates de qualidade entre cada fase.

---

## O que é este SDLC?

Este fluxo conecta os **9 agentes especializados** do framework em uma sequência de fases com:

- **Entradas e saídas** definidas por fase
- **Gates de qualidade** (DoR e DoD) antes de avançar
- **Artefatos** produzidos em cada etapa
- **Prompts prontos** para ativar cada fase com o Copilot

---

## Visão Geral das Fases

```
FASE 1 → FASE 2 → FASE 3 ──────────────────┐
Inception  Design  Development              │
                       ↕ (inner loop)       │
               FASE 4 ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
               Quality Gate
                    │
               FASE 5
               Release
                    │
               FASE 6
               Operate
                    │
                    └──► FASE 3 (próxima feature)
```

---

## Fases e Agentes Envolvidos

| Fase | Nome | Agentes | Artefato de Saída |
|------|------|---------|-------------------|
| 1 | [Inception](phases/01-inception.md) | Agent 01 | Discovery Report + Definition of Ready |
| 2 | [Design](phases/02-design.md) | Agent 02, 03, 08 | Architecture Document + Domain Model + Schema |
| 3 | [Development](phases/03-development.md) | Agent 04, 09 | Código revisado + testes passando |
| 4 | [Quality Gate](phases/04-quality-gate.md) | Agent 05, 06, 09 | Relatório de qualidade + aprovação |
| 5 | [Release](phases/05-release.md) | Agent 07 | Artefato deployado + changelog |
| 6 | [Operate](phases/06-operate.md) | Agent 07 | Runbook + alertas + post-mortem |

---

## Como Usar

### Iniciando um projeto novo

```bash
# Fase 1 — Discovery
@workspace #file:.github/copilot/sdlc/phases/01-inception.md
Quero criar [descrição do projeto]

# Fase 2 — Design (após Discovery Report aprovado)
@workspace #file:.github/copilot/sdlc/phases/02-design.md
#file:.github/copilot/skills/hexagonal-architecture.md
[Cole o Discovery Report]

# Fase 3 — Development (após Architecture Document aprovado)
@workspace #file:.github/copilot/sdlc/phases/03-development.md
#file:.github/copilot/agents/04-developer.md
#file:.github/copilot/languages/[linguagem]/profile.md
[Cole os arquivos relevantes da feature]

# Fase 4 — Quality Gate (antes de qualquer merge para main)
@workspace #file:.github/copilot/sdlc/phases/04-quality-gate.md
#file:.github/copilot/agents/05-code-reviewer.md
#file:.github/copilot/agents/06-test-engineer.md
[Cole os arquivos da feature]

# Fase 5 — Release
@workspace #file:.github/copilot/sdlc/phases/05-release.md
#file:.github/copilot/agents/07-devops.md

# Fase 6 — Operate (pós-deploy)
@workspace #file:.github/copilot/sdlc/phases/06-operate.md
```

---

## Definições Globais

### Definition of Ready (DoR) — Critérios para iniciar desenvolvimento

Uma história/feature está **pronta para desenvolvimento** quando:
- [ ] Requisitos escritos em linguagem de negócio (não técnica)
- [ ] Critérios de aceitação definidos e testáveis
- [ ] Dependências externas identificadas
- [ ] Estimativa de esforço realizada
- [ ] Protótipo/wireframe disponível (para features de UI)
- [ ] Discovery Report aprovado pelo time

### Definition of Done (DoD) — Critérios para considerar done

Uma história/feature está **concluída** quando:
- [ ] Código revisado por pelo menos 1 peer (Agent 05)
- [ ] Testes unitários e de integração passando
- [ ] Cobertura de código ≥ 80% no código novo
- [ ] Zero findings críticos ou altos na security review
- [ ] Documentação atualizada (ADRs, README, API docs)
- [ ] Feature deployada em staging e smoke test passando
- [ ] Aceita pelo Product Owner com os critérios de aceitação

---

## Gates de Qualidade

| Gate | Quando | Bloqueador |
|------|--------|-----------|
| **DoR Check** | Antes de iniciar Fase 3 | Sem Discovery Report aprovado, não coda |
| **Architecture Review** | Fim da Fase 2 | Sem ADRs documentados, não implementa |
| **Code Review** | Durante Fase 3, antes do merge | Finding crítico bloqueia o PR |
| **Security Review** | Fase 4, antes do release | Vulnerabilidade crítica bloqueia o deploy |
| **Quality Gate** | Fim da Fase 4 | Cobertura < 80% bloqueia o release |
| **Smoke Test** | Pós-deploy (Fase 5) | Falha dispara rollback automático |

---

## Referência de Arquivos

```
.github/copilot/sdlc/
├── README.md           ← Este arquivo
├── overview.md         ← Diagrama detalhado do fluxo
└── phases/
    ├── 01-inception.md
    ├── 02-design.md
    ├── 03-development.md
    ├── 04-quality-gate.md
    ├── 05-release.md
    └── 06-operate.md
```
