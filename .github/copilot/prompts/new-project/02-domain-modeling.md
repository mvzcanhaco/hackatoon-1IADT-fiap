# Prompt: Modelagem de Domínio (Novo Projeto)

## Como Usar

```
@workspace #file:.github/copilot/prompts/new-project/02-domain-modeling.md
#file:.github/copilot/skills/domain-driven-design.md

[cole o Discovery Report do passo anterior]

Modele o domínio deste sistema.
```

---

## Prompt Completo

Com base no Discovery Report fornecido, execute a **modelagem completa do domínio** seguindo DDD.

### Passo 1 — Event Storming

Mapeie todos os eventos de domínio em ordem cronológica:

```
FORMATO: [Evento] → desencadeia → [Próximo Evento ou Comando]

Exemplo:
VideoUploadado → [sistema detecta] → FramesExtracted
FramesExtracted → [modelo analisa] → ThreatDetected | NoThreatFound
ThreatDetected → [use case notifica] → AlertSent
```

Categorize cada evento:
- 🟧 **Event** (algo que aconteceu — passado)
- 🔵 **Command** (intenção de fazer algo — imperativo)
- 🟨 **Read Model / Query** (dados lidos)
- 🟪 **Aggregate** (onde o evento é processado)
- 🟦 **External System** (integração)

---

### Passo 2 — Identificar Bounded Contexts

Para cada contexto identificado, defina:

```markdown
### Bounded Context: [Nome]

**Responsabilidade central**: [uma frase]

**Pertence a este context**:
- Entidades: [liste]
- Eventos: [liste]
- Comandos: [liste]
- Regras de negócio: [liste]

**Integração com outros contexts**:
- [Context X] via [Shared Kernel / ACL / Customer-Supplier]
- Razão: [explique]
```

---

### Passo 3 — Modelar cada Context

Para cada bounded context, defina os building blocks:

#### Entidades
```
Para cada entidade:
- Nome: [NomeDaEntidade]
- Identidade: [campo que a identifica unicamente]
- Atributos: [lista]
- Comportamentos (verbos de negócio): [lista de métodos]
- Invariantes: [regras que nunca podem ser violadas]
- Eventos emitidos: [lista]
```

#### Value Objects
```
Para cada VO candidato:
- Nome: [NomeDoVO]
- Representa: [o que no domínio]
- Composto por: [atributos]
- Validações: [regras de validade]
- Imutável: sim
```

Candidatos típicos: IDs, Email, CPF/CNPJ, Money, Address, Phone, DateRange, Percentage, Status/State enums

#### Aggregates
```
Para cada aggregate:
- Aggregate Root: [nome da entidade raiz]
- Membros: [entidades e VOs que fazem parte]
- Invariantes do aggregate: [regras que o root garante para todos os membros]
- Fronteira de transação: [esta é a unidade que salva/carrega junto]
```

#### Domain Services
```
Quando a lógica não pertence a uma única entidade:
- Nome: [NomeDoService]
- Operação: [o que calcula/valida]
- Entidades envolvidas: [lista]
- Razão para ser service (não entity): [explique]
```

#### Repositories (Interfaces)
```
Para cada aggregate root:
- Interface: [NomeAggregate]Repository
- Operações: find_by_id, save, [outras queries específicas do domínio]
```

---

### Passo 4 — Ubiquitous Language Final

Gere o glossário completo:

```markdown
| Termo | Tipo | Definição no contexto do sistema |
|-------|------|----------------------------------|
| [Termo] | Entidade / VO / Evento / Comando / Regra | [definição precisa] |
```

---

## Output Esperado

```markdown
# Domain Model — [Nome do Sistema]

## Event Storming
[linha do tempo dos eventos]

## Bounded Contexts

### Context: [Nome]
- Responsabilidade: ...
- Integração: ...

## Modelo por Context

### Context: [Nome]

#### Entities
| Entidade | Identidade | Atributos | Comportamentos | Invariantes |
|----------|------------|-----------|----------------|-------------|
| [Nome] | [campo] | [lista] | [métodos] | [regras] |

#### Value Objects
| VO | Representa | Validações |
|----|------------|------------|
| [Nome] | ... | ... |

#### Aggregates
| Aggregate Root | Membros | Invariantes Garantidas |
|----------------|---------|------------------------|
| [Nome] | [lista] | [lista] |

#### Domain Events
| Evento | Produzido por | Quando | Dados |
|--------|---------------|--------|-------|
| [EventoNomeado] | [Entidade.método()] | [condição] | [campos] |

## Ubiquitous Language Completa
[tabela completa]

## Diagrama de Relacionamentos
[diagrama ASCII ou descrição]

## Próximo Passo
→ `#file:.github/copilot/prompts/new-project/03-architecture-design.md`
```

---

## Próximo Passo

```
@workspace #file:.github/copilot/prompts/new-project/03-architecture-design.md
[cole o Domain Model aqui]
Desenhe a arquitetura hexagonal para este domínio.
```
