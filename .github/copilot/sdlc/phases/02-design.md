# Fase 2: Design

## Objetivo

Transformar o Discovery Report em decisões técnicas concretas: arquitetura hexagonal, modelo de domínio, schema de banco e ADRs documentados.

**Duração típica**: 1-3 dias  
**Agentes**: `02-architect`, `03-domain-modeler`, `08-data-architect`  
**Participantes**: Tech Lead, Senior Dev, Domain Expert

---

## Entrada

- Discovery Report aprovado (Fase 1)
- Stack tecnológica definida

---

## Como Ativar

```
@workspace #file:.github/copilot/sdlc/phases/02-design.md
           #file:.github/copilot/agents/02-architect.md
           #file:.github/copilot/skills/hexagonal-architecture.md
           #file:.github/copilot/skills/domain-driven-design.md

[Cole o Discovery Report completo]
```

---

## Atividades da Fase

### 2.1 — Arquitetura com Agent 02

```
@workspace #file:.github/copilot/agents/02-architect.md
[Discovery Report]

Passo 1: Definir bounded contexts e seus relacionamentos
Passo 2: Definir estrutura de diretórios por camada
Passo 3: Especificar ports (interfaces) de cada contexto
Passo 4: Documentar ADRs para decisões arquiteturais
```

**Decisões mínimas a documentar como ADR**:
- Qual framework HTTP (FastAPI, Axum, Express, Spring)?
- Qual banco de dados e ORM?
- Como os bounded contexts se comunicam (sync/async)?
- Estratégia de autenticação?

### 2.2 — Modelagem de Domínio com Agent 03

```
@workspace #file:.github/copilot/agents/03-domain-modeler.md
[Discovery Report + Architecture Document]

Para cada bounded context:
- Identifique entidades e seus comportamentos
- Identifique value objects e suas invariantes
- Defina os aggregates e fronteiras de consistência
- Liste os domain events
- Documente o ubiquitous language
```

### 2.3 — Schema de Banco com Agent 08

```
@workspace #file:.github/copilot/agents/08-data-architect.md
[Domain Model]

Para cada aggregate:
- Mapeie para tabelas SQL
- Defina índices baseados nos padrões de acesso
- Escreva a migration inicial
- Identifique riscos de N+1
```

---

## Saídas (Artefatos)

### Architecture Design Document

```markdown
# Architecture Design Document — [Nome]

**Data**: [data] | **Versão**: 1.0

## 1. Bounded Contexts

### Contexto: [Nome]
- **Responsabilidade**: [o que este contexto controla]
- **Aggregates**: [lista]
- **Eventos produzidos**: [lista]
- **Eventos consumidos**: [lista]
- **Integrações**: [como se comunica com outros contextos]

## 2. Estrutura de Diretórios

```
[linguagem]/
├── domain/
│   ├── entities/
│   ├── value_objects/
│   ├── repositories/    ← ports
│   └── exceptions/
├── application/
│   ├── use_cases/
│   └── dtos/
├── infrastructure/
│   ├── repositories/
│   └── gateways/
└── presentation/
    └── http/
```

## 3. Ports Definidos

| Port | Tipo | Implementações previstas |
|------|------|--------------------------|
| [PortName] | Secundário | [tecnologia] |

## 4. ADRs

### ADR-001: [Título da Decisão]
**Contexto**: [por que esta decisão precisou ser tomada]
**Decisão**: [o que foi decidido]
**Consequências**: [o que muda, trade-offs]
**Alternativas rejeitadas**: [outras opções consideradas]
```

---

## Gate: Architecture Review

Checklist antes de avançar para a Fase 3:

- [ ] Domain Model com entidades, VOs e aggregates definidos
- [ ] Estrutura de diretórios criada (ou ao menos especificada)
- [ ] Ports (interfaces) especificadas para cada dependência externa
- [ ] ADRs para decisões chave (mínimo: framework, banco, autenticação)
- [ ] Schema de banco inicial escrito
- [ ] Domínio sem dependências de framework documentado como regra
- [ ] Bounded contexts com fronteiras claras
- [ ] Revisado por pelo menos 2 membros do time

**Sem Architecture Review aprovada, não inicie implementação.**

---

## Próxima Fase

→ [Fase 3: Development](03-development.md) com Architecture Document e Domain Model em mãos.
