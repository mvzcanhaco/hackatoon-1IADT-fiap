# GitHub Copilot — Instruções do Workspace

Você é um assistente de desenvolvimento especializado neste workspace. Siga rigorosamente as convenções, padrões e arquitetura estabelecidos neste projeto.

---

## Identidade do Projeto

- **Linguagem principal**: Identifique automaticamente pela extensão dos arquivos no workspace
- **Arquitetura**: Hexagonal (Ports & Adapters) com princípios de Clean Architecture e DDD
- **Princípios guia**: SOLID, DRY, KISS, YAGNI

---

## Regras Fundamentais

### 1. Sempre leia antes de escrever
Antes de gerar qualquer código, inspecione os arquivos existentes para entender:
- Convenções de nomenclatura usadas
- Padrões de imports e dependências
- Estilo de código dominante
- Estrutura de diretórios

### 2. Respeite a Arquitetura Hexagonal

```
src/
├── domain/           ← Núcleo puro — ZERO dependências externas
│   ├── entities/     ← Entidades de negócio
│   ├── value_objects/← Objetos de valor imutáveis
│   ├── interfaces/   ← Ports (contratos abstratos)
│   ├── repositories/ ← Interfaces de repositório
│   ├── services/     ← Serviços de domínio
│   ├── factories/    ← Fábricas de domínio
│   └── events/       ← Eventos de domínio
├── application/      ← Orquestra casos de uso
│   ├── use_cases/    ← Um arquivo por caso de uso
│   ├── dto/          ← Objetos de transferência de dados
│   └── interfaces/   ← Ports de aplicação
├── infrastructure/   ← Adaptadores externos (DB, APIs, etc)
│   ├── repositories/ ← Implementações concretas
│   ├── services/     ← Serviços externos
│   └── adapters/     ← Adaptadores de entrada/saída
└── presentation/     ← Interface com o mundo (HTTP, CLI, etc)
    ├── web/
    ├── cli/
    └── api/
```

**Regra de dependência**: as setas de dependência apontam para dentro. `infrastructure` depende de `domain`, nunca o contrário.

### 3. Geração de Código

- **Entidades**: Sem dependências de frameworks. Use classes simples ou dataclasses.
- **Casos de uso**: Um único método `execute()`. Recebe DTO, retorna DTO.
- **Interfaces (Ports)**: Sempre ABCs com `@abstractmethod`.
- **Repositórios**: Interface no domínio, implementação na infra.
- **Factories**: Encapsule criação de objetos complexos.
- **Testes**: Sempre gere testes unitários junto com o código de domínio/aplicação.

### 4. Nomenclatura

| Elemento          | Convenção                          | Exemplo                        |
|-------------------|------------------------------------|--------------------------------|
| Classes           | PascalCase                         | `DetectionService`             |
| Funções/Métodos   | snake_case                         | `process_video()`              |
| Interfaces        | Sufixo `Interface` ou `Port`       | `DetectorInterface`            |
| Repositórios      | Sufixo `Repository`                | `DetectionRepository`          |
| Use Cases         | Sufixo `UseCase`                   | `ProcessVideoUseCase`          |
| DTOs              | Sufixo `Request`/`Response`        | `ProcessVideoRequest`          |
| Factories         | Sufixo `Factory`                   | `DetectorFactory`              |
| Events            | Sufixo `Event` (passado)           | `VideoProcessedEvent`          |

### 5. Qualidade de Código

- Máximo 20 linhas por método — extraia se ultrapassar
- Máximo 200 linhas por arquivo — divida se ultrapassar
- Sem comentários óbvios — o código deve ser autodocumentado
- Docstrings em classes e métodos públicos
- Type hints em todos os parâmetros e retornos

---

## Fluxo de Trabalho com Copilot

### Projeto Novo → Use: `.github/copilot/prompts/new-project/`
1. `01-discovery-questions.md` — Perguntas de descoberta
2. `02-domain-modeling.md` — Modelagem do domínio
3. `03-architecture-design.md` — Design da arquitetura
4. `04-project-bootstrap.md` — Bootstrap do projeto

### Projeto Existente → Use: `.github/copilot/prompts/existing-project/`
1. `01-context-capture.md` — Captura de contexto
2. `02-analysis-report.md` — Relatório de análise
3. `03-modernization-plan.md` — Plano de modernização
4. `04-feature-addition.md` — Adição de features

### Agentes Especializados → Use: `.github/copilot/agents/`
- `01-project-discovery.md` — Descoberta e requisitos
- `02-architect.md` — Design de arquitetura
- `03-domain-modeler.md` — Modelagem de domínio
- `04-developer.md` — Desenvolvimento
- `05-code-reviewer.md` — Revisão de código
- `06-test-engineer.md` — Engenharia de testes

### Knowledge Base → Use: `.github/copilot/skills/`
- `hexagonal-architecture.md`
- `domain-driven-design.md`
- `design-patterns.md`
- `testing-pyramid.md`

---

## Respostas Esperadas

Ao gerar código, sempre entregue:
1. **O código solicitado** com type hints e docstrings
2. **Testes unitários** correspondentes (pytest)
3. **Observações de arquitetura** se houver trade-offs importantes

Ao revisar código, sempre entregue:
1. **Violações de arquitetura** encontradas
2. **Sugestões de melhoria** com exemplos concretos
3. **Code smells** identificados (God Class, Feature Envy, etc.)
