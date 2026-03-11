# Copilot Dev Framework v2.0

Framework de desenvolvimento para GitHub Copilot com **Protocolo GROUND-TRUTH anti-alucinação**, arquitetura hexagonal, DDD, suporte multi-linguagem e boas práticas de engenharia de software.

---

## Novidades v2.0

- ✅ **Protocolo GROUND-TRUTH** — regras anti-alucinação com marcadores de confiança
- ✅ **5 perfis de linguagem** — Python, Go, Android/Kotlin, C, C++
- ✅ **Detecção automática** de linguagem pelo workspace
- ✅ **Guardrails** — padrão de resposta com declaração de confiança

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
    ├── languages/                       ← 🆕 NOVO — Perfis por linguagem
    │   ├── python/
    │   │   └── profile.md               ← PEP 8/484/257, pyproject.toml, pytest
    │   ├── go/
    │   │   └── profile.md               ← Effective Go, table tests, project layout
    │   ├── android-kotlin/
    │   │   └── profile.md               ← MAD, Compose, MVVM, Hilt, Coroutines
    │   ├── c/
    │   │   └── profile.md               ← MISRA C, NASA JPL, Power of Ten Rules
    │   └── cpp/
    │       └── profile.md               ← C++ Core Guidelines, RAII, smart pointers
    │
    ├── agents/                          ← Personas especializadas
    │   ├── 01-project-discovery.md      ← Entrevista de requisitos (5 seções)
    │   ├── 02-architect.md              ← Design de arquitetura hexagonal
    │   ├── 03-domain-modeler.md         ← DDD: entidades, VOs, aggregates
    │   ├── 04-developer.md              ← Implementação inside-out
    │   ├── 05-code-reviewer.md          ← Review em 5 camadas
    │   └── 06-test-engineer.md          ← Pirâmide de testes
    │
    ├── skills/                          ← Knowledge base
    │   ├── hexagonal-architecture.md
    │   ├── domain-driven-design.md
    │   ├── design-patterns.md
    │   └── testing-pyramid.md
    │
    ├── prompts/
    │   ├── new-project/                 ← Fluxo para projetos novos
    │   │   ├── 01-discovery-questions.md
    │   │   ├── 02-domain-modeling.md
    │   │   ├── 03-architecture-design.md
    │   │   └── 04-project-bootstrap.md
    │   └── existing-project/            ← Fluxo para projetos existentes
    │       ├── 01-context-capture.md
    │       ├── 02-analysis-report.md
    │       ├── 03-modernization-plan.md
    │       └── 04-feature-addition.md
    │
    └── templates/
        ├── hexagonal/                   ← Templates de código por camada
        │   ├── domain/entity.md
        │   ├── domain/value-object.md
        │   ├── application/use-case.md
        │   ├── infrastructure/repository.md
        │   └── presentation/http-controller.md
        └── patterns/catalog.md          ← 12 Design Patterns com templates
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
- Linter da linguagem do seu projeto
