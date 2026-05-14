# GitHub Copilot — Instruções do Workspace

> **Versão**: 2.2 | **Protocolo**: GROUND-TRUTH ativo

---

## PROTOCOLO GROUND-TRUTH — Anti-Alucinação

**REGRA ABSOLUTA N°1**: Você NUNCA inventa. Você ancora.

Antes de gerar qualquer código:
1. **Leia os arquivos relevantes** com `#file:` — nunca assuma conteúdo
2. **Verifique a versão da linguagem/libs** nos arquivos de configuração
3. **Declare explicitamente** o que você não sabe

```
FONTES DE VERDADE ACEITAS:
  ✅ Código real do workspace (lido via #file:)
  ✅ Especificação oficial da linguagem (docs canônicos)
  ✅ Padrão reconhecido da comunidade (PEP, RFC, Core Guidelines)
  ❌ Suposição sem fonte verificável
  ❌ Memória sobre APIs sem confirmação no projeto
```

Referência completa: `#file:.github/copilot/guardrails/anti-hallucination.md`

**Gatilhos de parada obrigatória** — pare e pergunte quando:
- Não encontrou o arquivo referenciado
- Há inconsistência entre arquivos existentes
- A versão da lib pode não suportar a API sugerida
- A mudança pode quebrar código não fornecido no contexto

---

## Detecção Automática de Linguagem

Copilot identifica o perfil da linguagem pelos arquivos do workspace:

| Arquivos detectados | Perfil ativo |
|---------------------|-------------|
| `.py`, `pyproject.toml`, `requirements.txt` | → `#file:.github/copilot/languages/python/profile.md` |
| `.go`, `go.mod`, `go.sum` | → `#file:.github/copilot/languages/go/profile.md` |
| `AndroidManifest.xml`, `.kt`, `build.gradle` | → `#file:.github/copilot/languages/android-kotlin/profile.md` |
| `.c`, `.h` (sem `.cpp`) | → `#file:.github/copilot/languages/c/profile.md` |
| `.cpp`, `.cc`, `.hpp`, `CMakeLists.txt` c/ CXX | → `#file:.github/copilot/languages/cpp/profile.md` |
| `tsconfig.json`, `.ts`, `package.json` c/ `typescript` | → `#file:.github/copilot/languages/typescript/profile.md` |
| `pom.xml`, `build.gradle`, `.java` | → `#file:.github/copilot/languages/java/profile.md` |
| `Cargo.toml`, `.rs` | → `#file:.github/copilot/languages/rust/profile.md` |
| `composer.json`, `artisan`, `.php` | → `#file:.github/copilot/languages/php/profile.md` |

**Sempre carregue o perfil da linguagem** antes de implementar código:
```
@workspace #file:.github/copilot/languages/[linguagem]/profile.md
#file:.github/copilot/guardrails/anti-hallucination.md
[sua pergunta]
```

---

## Arquitetura — Regra Universal

Independente da linguagem, a regra de dependências é sempre a mesma:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Presentation │────▶│ Application  │────▶│   Domain     │◀────│Infrastructure│
│  (UI/API)   │     │  (UseCases)  │     │  (Entities)  │     │ (DB/HTTP/...) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
    entrada              orquestra           regras puras          implementa ports
```

**Regra de dependência**: setas apontam para dentro. Domínio não depende de ninguém.

### Estrutura por Linguagem

**Python** (`src/`):
```
domain/ → application/ → infrastructure/ → presentation/
```

**Go** (`internal/`):
```
domain/ → application/usecase/ → infrastructure/ → cmd/
```

**Android/Kotlin**:
```
domain/ → data/ → presentation/viewmodel/ → presentation/ui/
```

**C/C++** (`src/`):
```
domain/ → application/ → infrastructure/ → main
```

**TypeScript/Node.js** (`src/`):
```
domain/ → application/use-cases/ → infrastructure/ → presentation/http/
```

**Java/Spring Boot** (`src/main/java/`):
```
domain/ → application/usecase/ → infrastructure/persistence/ → presentation/http/
```

**Rust** (`src/`):
```
domain/ → application/use_cases/ → infrastructure/ → presentation/http/
```

**PHP/Laravel** (`app/`):
```
Domain/ → Application/UseCases/ → Infrastructure/Persistence/ → Http/Controllers/
```

---

## Convenções de Nomenclatura — Por Linguagem

| Elemento | Python | Go | Kotlin | C | C++ | TypeScript |
|----------|--------|----|--------|---|-----|------------|
| Classe/Tipo | `PascalCase` | `PascalCase` | `PascalCase` | `PascalCase` typedef | `PascalCase` | `PascalCase` |
| Função/Método | `snake_case` | `camelCase` | `camelCase` | `mod_verb_noun` | `camelCase`* | `camelCase` |
| Variável | `snake_case` | `camelCase` | `camelCase` | `snake_case` | `trailing_` membro | `camelCase` |
| Constante | `UPPER_SNAKE` | `kPascalCase` | `UPPER_SNAKE` | `UPPER_SNAKE` | `kPascalCase` | `UPPER_SNAKE` |
| Interface | `*Port`/`*Interface` | `-er` sufixo | `*Interface` | typedef struct | classe abstrata | `*Repository`/`*Gateway` |
| Privado | `_prefixo` | unexported | `private` | `static` | `trailing_` | `private` keyword |

*C++: verifique o estilo adotado no projeto antes de escolher — respeite a convenção existente.

---

## Regras de Geração de Código

### 1. Leia antes de escrever
```
❌ NUNCA assuma o conteúdo de um arquivo sem lê-lo
✅ SEMPRE referencie com #file: arquivos que serão modificados
✅ SEMPRE verifique dependências existentes antes de sugerir novas
```

### 2. Marcadores de confiança obrigatórios
```
[VERIFIED: arquivo:linha]   — verificado no workspace
[STANDARD: PEP8/RFC/guide] — padrão oficial da linguagem
[ASSUMPTION: motivo]        — suposição que precisa de validação humana
[PLACEHOLDER]               — substituir pelo valor real do projeto
[TODO: verify API version]  — verificar compatibilidade de versão
```

### 3. Limites de qualidade por linguagem

| Métrica | Python | Go | Kotlin | C | C++ | TypeScript |
|---------|--------|----|----|---|-----|------------|
| Máx. linhas/função | 20 | 40 | 30 | 60 | 40 | 30 |
| Máx. linhas/arquivo | 300 | 500 | 400 | 500 | 500 | 400 |
| Type safety | mypy strict | interfaces | type system | stdint.h | const+[[nodiscard]] | strict mode + noUncheckedIndexedAccess |
| Linter | ruff (E,F,I,B,UP,SIM,PERF)+mypy | golangci-lint | ktlint+detekt | clang-tidy | clang-tidy | ESLint+typescript-eslint strict |
| Env manager | uv | Go modules | Gradle | Make/CMake | CMake | npm / pnpm |

### 4. Testes sempre junto com o código

Para cada implementação entregue, inclua:
- Unitários do comportamento de domínio
- Mocks/stubs para dependências externas
- Pelo menos 1 caso de erro/falha

---

## Fluxo de Trabalho com Copilot

### Projeto Novo (qualquer linguagem)
```
1: #file:.github/copilot/prompts/new-project/01-discovery-questions.md
2: #file:.github/copilot/prompts/new-project/02-domain-modeling.md
3: #file:.github/copilot/prompts/new-project/03-architecture-design.md
4: #file:.github/copilot/prompts/new-project/04-project-bootstrap.md
5: #file:.github/copilot/prompts/new-project/05-security-threat-model.md   ← STRIDE, OWASP, LGPD
6: #file:.github/copilot/prompts/new-project/06-api-contract-design.md     ← Design-first OpenAPI
```

### Projeto Existente
```
1: #file:.github/copilot/prompts/existing-project/01-context-capture.md
2: #file:.github/copilot/prompts/existing-project/02-analysis-report.md
3: #file:.github/copilot/prompts/existing-project/03-modernization-plan.md
4: #file:.github/copilot/prompts/existing-project/04-feature-addition.md
5: #file:.github/copilot/prompts/existing-project/05-performance-audit.md  ← N+1, cache, índices
```

### SDLC Completo (Discovery → Produção)
```
Visão geral:  #file:.github/copilot/sdlc/README.md
Diagrama:     #file:.github/copilot/sdlc/overview.md

Fase 1 — Inception:      #file:.github/copilot/sdlc/phases/01-inception.md
Fase 2 — Design:         #file:.github/copilot/sdlc/phases/02-design.md
Fase 3 — Development:    #file:.github/copilot/sdlc/phases/03-development.md
Fase 4 — Quality Gate:   #file:.github/copilot/sdlc/phases/04-quality-gate.md
Fase 5 — Release:        #file:.github/copilot/sdlc/phases/05-release.md
Fase 6 — Operate:        #file:.github/copilot/sdlc/phases/06-operate.md
```

### Agentes Especializados
| Tarefa | Agent |
|--------|-------|
| Levantamento de requisitos | `#file:.github/copilot/agents/01-project-discovery.md` |
| Design arquitetural | `#file:.github/copilot/agents/02-architect.md` |
| Modelagem de domínio | `#file:.github/copilot/agents/03-domain-modeler.md` |
| Implementação | `#file:.github/copilot/agents/04-developer.md` |
| Code review | `#file:.github/copilot/agents/05-code-reviewer.md` |
| Geração de testes | `#file:.github/copilot/agents/06-test-engineer.md` |
| CI/CD e infraestrutura | `#file:.github/copilot/agents/07-devops.md` |
| Schema de banco + migrations | `#file:.github/copilot/agents/08-data-architect.md` |
| Revisão de segurança + OWASP | `#file:.github/copilot/agents/09-security-engineer.md` |

### Skills — Knowledge Base
| Tópico | Skill |
|--------|-------|
| Arquitetura hexagonal | `#file:.github/copilot/skills/hexagonal-architecture.md` |
| Domain-Driven Design | `#file:.github/copilot/skills/domain-driven-design.md` |
| Design Patterns (12) | `#file:.github/copilot/skills/design-patterns.md` |
| Pirâmide de testes | `#file:.github/copilot/skills/testing-pyramid.md` |
| Observabilidade (OTEL, Prometheus, Grafana) | `#file:.github/copilot/skills/observability.md` |
| Design de APIs REST + OpenAPI | `#file:.github/copilot/skills/api-design.md` |
| Event-Driven (Kafka, Outbox, Saga) | `#file:.github/copilot/skills/event-driven.md` |
| Caching (Redis, Cache-Aside, Rate Limiting) | `#file:.github/copilot/skills/caching.md` |

---

## Respostas Esperadas

### Ao gerar código:
1. Código com marcadores de confiança (`[VERIFIED]`, `[ASSUMPTION]`)
2. Testes correspondentes
3. **Declaração de confiança** ao final:
   ```
   ✅ Alto: [partes verificadas em fontes concretas]
   ⚠️ Médio: [baseadas em padrão, não verificadas no projeto]
   ❌ Baixo: [especulativas — revisar antes de usar em produção]
   ```

### Ao revisar código:
1. Violações por severidade (🔴 Crítico → 🔵 Baixo)
2. Código atual vs. código sugerido lado a lado
3. Referência ao padrão violado (PEP, Core Guidelines, etc.)

### Quando não souber:
```
"Não encontrei [X] nos arquivos fornecidos.
Para responder com precisão, preciso de: #file:[caminho]
O que posso afirmar com certeza: [apenas o que está documentado]"
```

---

## Checklist Universal Pré-Entrega

- [ ] Todos os imports existem no projeto (verificado no arquivo de dependências)
- [ ] Tipos/assinaturas batem com definições reais (verificado via #file:)
- [ ] Versão da linguagem suporta as features usadas (verificado no config)
- [ ] Nenhuma lógica de negócio inventada sem especificação explícita
- [ ] Partes incertas marcadas com [ASSUMPTION] ou [TODO: verify]
- [ ] Testes cobrem happy path + pelo menos 1 caso de erro
- [ ] Sem lógica de negócio fora da camada de domínio
