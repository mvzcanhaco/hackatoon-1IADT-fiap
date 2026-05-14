# Copilot Dev Framework v2.2

Framework de desenvolvimento para GitHub Copilot com **Protocolo GROUND-TRUTH anti-alucinação**, arquitetura hexagonal, DDD, suporte multi-linguagem e boas práticas de engenharia de software.

---

## Novidades v2.2

- ✅ **Protocolo GROUND-TRUTH** — regras anti-alucinação com marcadores de confiança
- ✅ **9 perfis de linguagem** — Python, Go, Android/Kotlin, C, C++, TypeScript, **Java, Rust, PHP**
- ✅ **9 agentes especializados** — inclui **Agent 08: Data Architect** e **Agent 09: Security Engineer**
- ✅ **SDLC completo** — 6 fases (Inception → Design → Dev → Quality Gate → Release → Operate)
- ✅ **4 Skills** — Observability, API Design, Event-Driven, Caching
- ✅ **6 Templates** — Domain Event, Port Interface, Exceptions, **Aggregate Root, Query Handler, Event Publisher**
- ✅ **6 Prompts** — 4 novos + **Security Threat Model, Performance Audit, API Contract Design**
- ✅ **Loop de revisão** — ciclo dev → review → fix → approve documentado no Agent 05
- ✅ **Testes E2E** — templates de ponta a ponta no Agent 06
- ✅ **Detecção automática** de linguagem pelo workspace
- ✅ **Guardrails** — padrão de resposta com declaração de confiança
- ✅ **ROADMAP** — versões passadas e planejamento futuro

---

## Como Usar

Este framework transforma o GitHub Copilot em um ecossistema completo de desenvolvimento. Use `#file:` para incluir qualquer documento como contexto.

```
@workspace #file:.github/copilot/guardrails/anti-hallucination.md
           #file:.github/copilot/languages/python/profile.md
           #file:.github/copilot/agents/04-developer.md
           #file:src/domain/entities/detection.py
           Implemente o use case ProcessVideoUseCase
```

---

## Estrutura Completa

```
.github/
├── copilot-instructions.md              ← Carregado AUTOMATICAMENTE (v2.0 + anti-alucinação)
└── copilot/
    ├── README.md                        ← Este arquivo
    │
    ├── guardrails/                      ← 🛡️ NOVO — Proteção contra alucinação
    │   └── anti-hallucination.md        ← Protocolo GROUND-TRUTH completo
    │
    ├── languages/                       ← Perfis por linguagem (9 linguagens)
    │   ├── python/profile.md            ← PEP 8/484/257, pyproject.toml, pytest
    │   ├── go/profile.md                ← Effective Go, table tests, project layout
    │   ├── android-kotlin/profile.md    ← MAD, Compose, MVVM, Hilt, Coroutines
    │   ├── c/profile.md                 ← MISRA C, NASA JPL, Power of Ten Rules
    │   ├── cpp/profile.md               ← C++ Core Guidelines, RAII, smart pointers
    │   ├── typescript/profile.md        ← TypeScript 5.x, Node.js 20 LTS, Zod, Vitest
    │   ├── java/profile.md              ← Java 21 LTS, Spring Boot 3.x, Records, Testcontainers
    │   ├── rust/profile.md              ← Rust 2021, tokio, axum, sqlx, thiserror
    │   └── php/profile.md               ← PHP 8.3, Laravel 11, Pest, readonly classes
    │
    ├── agents/                          ← Personas especializadas (9 agentes)
    │   ├── 01-project-discovery.md      ← Entrevista de requisitos (5 seções)
    │   ├── 02-architect.md              ← Design de arquitetura hexagonal
    │   ├── 03-domain-modeler.md         ← DDD: entidades, VOs, aggregates
    │   ├── 04-developer.md              ← Implementação inside-out (DTO vs Schema)
    │   ├── 05-code-reviewer.md          ← Review em 5 camadas + loop de feedback
    │   ├── 06-test-engineer.md          ← Pirâmide de testes + E2E templates
    │   ├── 07-devops.md                 ← CI/CD, Docker, observabilidade
    │   ├── 08-data-architect.md         ← ERD, schema, migrations, N+1, outbox
    │   └── 09-security-engineer.md      ← OWASP, STRIDE, LGPD, threat modeling
    │
    ├── skills/                          ← Knowledge base (8 skills)
    │   ├── hexagonal-architecture.md
    │   ├── domain-driven-design.md
    │   ├── design-patterns.md
    │   ├── testing-pyramid.md
    │   ├── observability.md             ← Logs, Métricas, Traces, OTEL, Prometheus
    │   ├── api-design.md               ← REST, OpenAPI 3.1, RFC 7807, paginação
    │   ├── event-driven.md             ← CloudEvents, Outbox, Saga, Kafka
    │   └── caching.md                  ← Cache-Aside, TTL, Rate Limiting, Redis
    │
    ├── prompts/
    │   ├── new-project/                 ← Fluxo para projetos novos
    │   │   ├── 01-discovery-questions.md
    │   │   ├── 02-domain-modeling.md
    │   │   ├── 03-architecture-design.md
    │   │   ├── 04-project-bootstrap.md
    │   │   ├── 05-security-threat-model.md  ← STRIDE, OWASP, LGPD
    │   │   └── 06-api-contract-design.md    ← Design-first OpenAPI
    │   └── existing-project/            ← Fluxo para projetos existentes
    │       ├── 01-context-capture.md
    │       ├── 02-analysis-report.md
    │       ├── 03-modernization-plan.md
    │       ├── 04-feature-addition.md
    │       └── 05-performance-audit.md  ← N+1, cache, índices, load test
    │
    ├── sdlc/                            ← Fluxo SDLC completo (6 fases)
    │   ├── README.md                    ← Visão geral e como usar
    │   ├── overview.md                  ← Diagrama completo com agentes e gates
    │   └── phases/
    │       ├── 01-inception.md          ← Discovery + DoR + Product Vision
    │       ├── 02-design.md             ← Arquitetura + DDD + Schema
    │       ├── 03-development.md        ← Inner loop inside-out + DoD
    │       ├── 04-quality-gate.md       ← Code review + tests + security gate
    │       ├── 05-release.md            ← Deploy + smoke tests + rollback
    │       └── 06-operate.md            ← SLOs + alertas + runbooks + post-mortem
    │
    ├── templates/
    │   ├── hexagonal/                   ← Templates de código por camada
    │   │   ├── domain/entity.md
    │   │   ├── domain/value-object.md
    │   │   ├── domain/aggregate-root.md ← Invariantes, factory, reconstitute, pull_events
    │   │   ├── domain/domain-event.md   ← Domain Events imutáveis
    │   │   ├── domain/port-interface.md ← Ports primários e secundários
    │   │   ├── domain/exceptions.md     ← Hierarquia de exceções de domínio
    │   │   ├── application/use-case.md
    │   │   ├── application/query-handler.md  ← CQRS read side, read model, paginação
    │   │   ├── infrastructure/repository.md
    │   │   ├── infrastructure/event-publisher.md  ← Kafka, Outbox, InMemory
    │   │   └── presentation/http-controller.md
    │   └── patterns/catalog.md          ← 12 Design Patterns com templates
    │
    └── ROADMAP.md                       ← Versões passadas e planejamento futuro
```

---

## Fluxos de Trabalho

### 🆕 Fluxo Base Recomendado (com anti-alucinação)

Inclua **sempre** o guardrail nas sessões de implementação:

```
@workspace #file:.github/copilot/guardrails/anti-hallucination.md
           #file:.github/copilot/languages/[linguagem]/profile.md
           #file:.github/copilot/agents/[agente].md
           [arquivos do projeto relevantes]
           [sua pergunta]
```

### Fluxo 1: Novo Projeto

```bash
# Passo 1 — descoberta de requisitos
@workspace #file:.github/copilot/prompts/new-project/01-discovery-questions.md
Quero criar [descrição do projeto]

# Passo 2 — modelagem de domínio
@workspace #file:.github/copilot/prompts/new-project/02-domain-modeling.md
[cole o Discovery Report do passo 1]

# Passo 3 — design de arquitetura
@workspace #file:.github/copilot/prompts/new-project/03-architecture-design.md
#file:.github/copilot/skills/hexagonal-architecture.md
[cole o Domain Model do passo 2]

# Passo 4 — scaffold do projeto
@workspace #file:.github/copilot/prompts/new-project/04-project-bootstrap.md
#file:.github/copilot/languages/[linguagem]/profile.md
[cole o Architecture Design do passo 3]
```

### Fluxo 2: Projeto Existente

```bash
# Passo 1 — captura de contexto (inclua os arquivos mais importantes)
@workspace #file:.github/copilot/prompts/existing-project/01-context-capture.md
#file:src/domain/entities/[entidade].py
#file:src/application/use_cases/[use_case].py
Analise o contexto deste projeto

# Passo 2 — análise aprofundada
@workspace #file:.github/copilot/prompts/existing-project/02-analysis-report.md
[cole o Context Capture Report]
Foco em: [arquitetura / qualidade / performance / segurança]

# Passo 3 — plano de modernização ou nova feature
@workspace #file:.github/copilot/prompts/existing-project/04-feature-addition.md
#file:.github/copilot/guardrails/anti-hallucination.md
#file:[arquivos relevantes para a feature]
Feature: [descrição]
```

---

## Perfis de Linguagem — Referência Rápida

### Python
```
@workspace #file:.github/copilot/languages/python/profile.md
Referências: PEP 8, PEP 484 (type hints), PEP 257 (docstrings)
Linter: ruff + mypy strict
Testes: pytest + coverage
Config: pyproject.toml
```

### Go
```
@workspace #file:.github/copilot/languages/go/profile.md
Referências: Effective Go, Go Code Review Comments, Standard Project Layout
Linter: golangci-lint
Testes: table-driven com t.Parallel()
Config: go.mod
```

### Android / Kotlin
```
@workspace #file:.github/copilot/languages/android-kotlin/profile.md
Referências: Modern Android Development (MAD), Android Architecture Guide
Stack: Compose + ViewModel + Hilt + Room + Coroutines + Flow
Testes: JUnit5 + MockK + Turbine
Config: build.gradle.kts + libs.versions.toml
```

### C
```
@workspace #file:.github/copilot/languages/c/profile.md
Referências: MISRA C:2012, NASA JPL Power of Ten, SEI CERT C
Padrão: C11 ou C17
Linter: clang-tidy + cppcheck
Flags: -Wall -Wextra -Werror -pedantic
```

### C++
```
@workspace #file:.github/copilot/languages/cpp/profile.md
Referências: C++ Core Guidelines (Stroustrup & Sutter)
Padrão: C++17 mínimo, C++20 preferível
Paradigma: RAII + smart pointers + Rule of Zero
Linter: clang-tidy + Address Sanitizer
```

### TypeScript / Node.js
```
@workspace #file:.github/copilot/languages/typescript/profile.md
Referências: TypeScript Handbook, Node.js Best Practices
Stack: TypeScript 5.x + Node.js 20 LTS + Zod + Vitest
Linter: ESLint (strict) + Prettier
Config: tsconfig.json (strict mode) + package.json
```

### Java / Spring Boot
```
@workspace #file:.github/copilot/languages/java/profile.md
Referências: Effective Java, Spring Boot Reference, Google Java Style
Versão: Java 21 LTS + Spring Boot 3.x
Padrões: Records, Sealed classes, pattern matching
Testes: JUnit 5 + Mockito + Testcontainers + AssertJ
```

### Rust
```
@workspace #file:.github/copilot/languages/rust/profile.md
Referências: The Rust Book, Rust API Guidelines
Stack: Rust 2021 + tokio + axum + sqlx + thiserror
Error handling: Result<T, E> com thiserror (libs) / anyhow (binários)
Testes: cargo test + mockall + proptest
```

### PHP / Laravel
```
@workspace #file:.github/copilot/languages/php/profile.md
Referências: PHP 8.3 Manual, Laravel 11 Docs, PHP-FIG PSR-12
Stack: PHP 8.3 + Laravel 11 + Pest PHP
Padrões: readonly classes, enums, strict_types
DI: Laravel Service Container com ServiceProvider
```

---

## Fluxo SDLC Completo

Para conduzir um projeto do zero ao ar, use o SDLC estruturado:

```bash
# Visão geral do processo
#file:.github/copilot/sdlc/README.md
#file:.github/copilot/sdlc/overview.md

# Fase 1 — Inception (Discovery + DoR)
@workspace #file:.github/copilot/sdlc/phases/01-inception.md
           #file:.github/copilot/agents/01-project-discovery.md

# Fase 2 — Design (Arquitetura + DDD + Schema)
@workspace #file:.github/copilot/sdlc/phases/02-design.md
           #file:.github/copilot/agents/02-architect.md
           #file:.github/copilot/agents/08-data-architect.md

# Fase 3 — Development (inner loop inside-out)
@workspace #file:.github/copilot/sdlc/phases/03-development.md
           #file:.github/copilot/agents/04-developer.md
           #file:.github/copilot/agents/09-security-engineer.md

# Fase 4 — Quality Gate (review + testes + segurança)
@workspace #file:.github/copilot/sdlc/phases/04-quality-gate.md
           #file:.github/copilot/agents/05-code-reviewer.md
           #file:.github/copilot/agents/06-test-engineer.md

# Fase 5 — Release (deploy + smoke tests + rollback)
@workspace #file:.github/copilot/sdlc/phases/05-release.md
           #file:.github/copilot/agents/07-devops.md

# Fase 6 — Operate (SLOs + alertas + runbooks + post-mortem)
@workspace #file:.github/copilot/sdlc/phases/06-operate.md
```

---

## Protocolo Anti-Alucinação — GROUND-TRUTH

### Os 3 pilares

```
PILAR 1 — ANCORAGEM
  Toda afirmação deve ter fonte verificável:
  workspace (#file:) | spec oficial | padrão reconhecido

PILAR 2 — MARCAÇÃO
  Código com confiança declarada:
  [VERIFIED] | [STANDARD] | [ASSUMPTION] | [PLACEHOLDER]

PILAR 3 — TRANSPARÊNCIA
  Declaração de confiança ao final de toda resposta com código
```

### Exemplo de resposta segura

```
## Código gerado

```python
# [VERIFIED: src/domain/interfaces/detector.py:15] — interface real
class ProcessVideoUseCase:
    def __init__(self, detector: DetectorInterface) -> None:  # [VERIFIED]
        self._detector = detector  # [STANDARD: PEP 8 private prefix]

    def execute(self, request: ProcessVideoRequest) -> ProcessVideoResponse:
        # [ASSUMPTION: ProcessVideoRequest tem campo video_path — confirmar]
        result = self._detector.process(request.video_path)  # [TODO: verify API]
        return ProcessVideoResponse(detections=result.detections)
```

## Declaração de Confiança
✅ Alto: assinatura de DetectorInterface (verificada no arquivo)
⚠️ Médio: padrão de use case (baseado no framework, não verificado no projeto)
❌ Baixo: campo video_path no Request — verificar definição real do DTO
```

---

## Instalação em Outro Projeto

```bash
# Copie a estrutura para o root do seu projeto
cp -r .github/copilot /seu-projeto/.github/copilot
cp .github/copilot-instructions.md /seu-projeto/.github/copilot-instructions.md
cp .editorconfig /seu-projeto/.editorconfig

# O Copilot carregará as instruções automaticamente no VS Code
```

**Extensões VS Code recomendadas**:
- `github.copilot` (obrigatório)
- `github.copilot-chat` (obrigatório)
- `ms-python.python` + `ms-python.mypy-type-checker` + `charliermarsh.ruff` → Python
- `golang.go` → Go
- `dbaeumer.vscode-eslint` + `esbenp.prettier-vscode` → TypeScript/Node.js
- `mathiasfrohlich.kotlin` + `android-studio` → Android/Kotlin
- `llvm-vs-code-extensions.vscode-clangd` → C/C++
- `ms-azuretools.vscode-docker` → Docker
- `github.vscode-github-actions` → GitHub Actions
