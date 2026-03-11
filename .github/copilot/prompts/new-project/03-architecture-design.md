# Prompt: Design de Arquitetura (Novo Projeto)

## Como Usar

```
@workspace #file:.github/copilot/prompts/new-project/03-architecture-design.md
#file:.github/copilot/skills/hexagonal-architecture.md
#file:.github/copilot/skills/design-patterns.md

[cole o Domain Model do passo anterior]

Projete a arquitetura hexagonal completa deste sistema.
```

---

## Prompt Completo

Com base no Domain Model fornecido, projete a **arquitetura hexagonal completa** do sistema.

### Fase 1 — Estrutura de Diretórios

Gere a estrutura completa de diretórios:

```
[nome-do-projeto]/
├── src/
│   ├── [context_1]/
│   │   ├── domain/
│   │   │   ├── entities/
│   │   │   ├── value_objects/
│   │   │   ├── repositories/     ← interfaces apenas
│   │   │   ├── services/
│   │   │   ├── events/
│   │   │   ├── factories/
│   │   │   └── exceptions/
│   │   ├── application/
│   │   │   ├── use_cases/        ← um arquivo por use case
│   │   │   ├── dto/
│   │   │   └── ports/            ← interfaces para serviços externos
│   │   ├── infrastructure/
│   │   │   ├── repositories/     ← implementações concretas
│   │   │   ├── adapters/
│   │   │   └── config/
│   │   └── presentation/
│   │       ├── http/             ← controllers REST
│   │       └── cli/              ← se aplicável
├── tests/
│   ├── unit/
│   │   ├── [context_1]/
│   │   │   ├── domain/
│   │   │   └── application/
│   ├── integration/
│   │   └── [context_1]/
│   │       └── infrastructure/
│   ├── e2e/
│   └── conftest.py
├── docs/
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── decisions/            ← ADRs
│   │   └── diagrams/
│   └── ubiquitous-language.md
├── [configurações do projeto]
└── README.md
```

---

### Fase 2 — Ports (Interfaces)

Para cada camada, liste todas as interfaces necessárias:

#### Ports de Saída (Driven Ports — Infra implementa)
```python
# No domínio ou aplicação

class [Aggregate]Repository(ABC):
    """O que o use case precisa do repositório."""
    @abstractmethod def find_by_id(self, id: [Id]) -> Optional[[Entity]]: ...
    @abstractmethod def save(self, entity: [Entity]) -> None: ...

class [ExternalService]Port(ABC):
    """O que o use case precisa do serviço externo."""
    @abstractmethod def [operation](self, [params]) -> [ReturnType]: ...
```

#### Ports de Entrada (Driving Ports — Use Cases implementam)
```python
# Definidos na aplicação — UI os chama

class [FeatureName]UseCase(Protocol):
    """Contrato que a apresentação usa."""
    def execute(self, request: [Request]) -> [Response]: ...
```

---

### Fase 3 — Use Cases

Liste todos os use cases e seus contratos:

```markdown
| Use Case | Input (Request DTO) | Output (Response DTO) | Regras Aplicadas |
|----------|--------------------|-----------------------|------------------|
| [UseCaseName] | [campos] | [campos] | [regras] |
```

Para cada use case, especifique o fluxo:
```
1. Recebe [RequestDTO]
2. Busca [Entity] via [Repository]
3. Executa [domain method]
4. Persiste via [Repository]
5. Notifica via [ServicePort] (se houver)
6. Retorna [ResponseDTO]
```

---

### Fase 4 — Adapters (Infraestrutura)

Liste os adapters necessários e suas tecnologias:

```markdown
| Adapter | Port que implementa | Tecnologia | Descrição |
|---------|--------------------|-----------|-----------
| [NomeRepository] | [RepoInterface] | SQLAlchemy / MongoDB / Redis | ... |
| [NomeServiceAdapter] | [ServicePort] | HTTP / SDK / gRPC | ... |
```

---

### Fase 5 — Decisões Arquiteturais (ADRs)

Documente as decisões mais importantes:

```markdown
### ADR-001: [Título]
- **Decisão**: ...
- **Justificativa**: ...
- **Alternativas rejeitadas**: ...
- **Consequências**: ...

### ADR-002: Estratégia de Persistência
### ADR-003: Comunicação entre Bounded Contexts
### ADR-004: Estratégia de Autenticação/Autorização
### ADR-005: Estratégia de Deploy e Infraestrutura
```

---

### Fase 6 — Design Patterns Recomendados

Para as situações identificadas no domínio, recomende patterns:

```markdown
| Situação | Pattern | Onde aplicar |
|----------|---------|--------------|
| [situação do domínio] | [Pattern] | [camada/arquivo] |
```

---

## Output Esperado

```markdown
# Architecture Design — [Nome do Sistema]

## Visão Geral
[diagrama ASCII da arquitetura]

## Bounded Contexts e Responsabilidades
[mapa de contexts]

## Estrutura de Diretórios
[árvore completa]

## Ports (Contratos)
[interfaces definidas]

## Use Cases
[tabela completa]

## Adapters
[tabela completa]

## Fluxo de Dados
[sequência das requisições mais importantes]

## ADRs
[decisões documentadas]

## Patterns Aplicados
[onde cada pattern será usado]

## Stack Tecnológica Recomendada
| Categoria | Tecnologia | Justificativa |
|-----------|-----------|---------------|
| Framework web | FastAPI / Django / Express | ... |
| ORM | SQLAlchemy / Prisma | ... |
| Banco de dados | PostgreSQL / MongoDB | ... |
| Testes | pytest / jest | ... |
| CI/CD | GitHub Actions | ... |

## Próximo Passo
→ `#file:.github/copilot/prompts/new-project/04-project-bootstrap.md`
```

---

## Próximo Passo

```
@workspace #file:.github/copilot/prompts/new-project/04-project-bootstrap.md
[cole o Architecture Design aqui]
Gere o scaffold completo do projeto.
```
