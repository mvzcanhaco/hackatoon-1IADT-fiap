# Fase 3: Development

## Objetivo

Implementar a feature de dentro para fora (inside-out): domínio → aplicação → infraestrutura → apresentação, com testes e revisão de segurança no código crítico.

**Duração típica**: 1-5 dias por feature  
**Agentes**: `04-developer`, `09-security-engineer`  
**Participantes**: Desenvolvedor(es)

---

## Entrada

- Architecture Design Document aprovado (Fase 2)
- Domain Model definido
- Feature branch criada a partir de `main`

---

## Como Ativar

```
@workspace #file:.github/copilot/sdlc/phases/03-development.md
           #file:.github/copilot/agents/04-developer.md
           #file:.github/copilot/guardrails/anti-hallucination.md
           #file:.github/copilot/languages/[linguagem]/profile.md
           #file:[arquivos existentes relevantes]

Feature: [descrição da feature]
Regras de negócio: [liste as regras do Discovery Report]
```

---

## Inner Loop do Desenvolvedor

```
┌─────────────────────────────────────────────────────┐
│                  INNER LOOP                          │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Escrever │→  │ Lint +   │→  │ Testes unitários │ │
│  │ código   │   │ Format   │   │ (deve passar)    │ │
│  └──────────┘   └──────────┘   └──────────────────┘ │
│       ▲                                 │            │
│       └─────────── fix ─────────────────┘            │
└─────────────────────────────────────────────────────┘
```

---

## Ordem de Implementação: Inside-Out

### Passo 1 — Domain Layer (sem framework)

```
@workspace #file:.github/copilot/agents/04-developer.md
#file:.github/copilot/templates/hexagonal/domain/entity.md
#file:.github/copilot/templates/hexagonal/domain/value-object.md
#file:.github/copilot/templates/hexagonal/domain/exceptions.md
#file:.github/copilot/templates/hexagonal/domain/port-interface.md

Implemente:
1. Value Objects: [lista de VOs]
2. Entidade: [EntityName] com comportamentos: [lista]
3. Exceções de domínio: [lista]
4. Ports (interfaces): [lista de repositories/gateways]
```

**Checklist Passo 1**:
- [ ] Entidades sem imports de framework
- [ ] Value Objects com validação no construtor
- [ ] Exceções de domínio criadas
- [ ] Testes unitários da entidade passando

### Passo 2 — Application Layer (Use Cases)

```
@workspace #file:.github/copilot/agents/04-developer.md
#file:.github/copilot/templates/hexagonal/application/use-case.md
#file:src/domain/entities/[entity].py
#file:src/domain/repositories/[repo].py

Implemente o use case [UseCaseName]:
- Input: [RequestDTO fields]
- Output: [ResponseDTO fields]
- Regras: [regras de negócio do Discovery Report]
```

**Checklist Passo 2**:
- [ ] Use case tem apenas um método `execute()`
- [ ] Recebe dependências via construtor
- [ ] Não faz chamadas HTTP ou banco diretamente
- [ ] Testes unitários com mocks passando

### Passo 3 — Infrastructure Layer

```
@workspace #file:.github/copilot/agents/04-developer.md
#file:.github/copilot/templates/hexagonal/infrastructure/repository.md
#file:src/domain/repositories/[repo_interface].py
#file:src/domain/entities/[entity].py

Implemente SQLAlchemy[Entity]Repository.
Mapeamento: [campos da entidade → colunas]
```

**Checklist Passo 3**:
- [ ] Implementa o port (interface) do domínio
- [ ] `_to_domain()` e `_to_persistence()` separados
- [ ] Nenhuma lógica de negócio
- [ ] InMemory Fake para testes
- [ ] Testes de integração com DB real passando

### Passo 4 — Presentation Layer

```
@workspace #file:.github/copilot/agents/04-developer.md
#file:.github/copilot/templates/hexagonal/presentation/http-controller.md
#file:src/application/use_cases/[use_case].py

Implemente o controller para [recurso]:
- Endpoints: [lista]
- Schemas de validação: [campos]
- Mapeamento de erros: [DomainException → HTTP status]
```

**Checklist Passo 4**:
- [ ] Schemas Pydantic/Zod validam entrada (não o domínio)
- [ ] Controller converte Schema → DTO → Use Case → DTO → Schema
- [ ] Todos os DomainExceptions mapeados para HTTP
- [ ] Docstrings OpenAPI nos endpoints

---

## Revisão de Segurança no Código Crítico

Para endpoints de autenticação, dados pessoais ou operações financeiras:

```
@workspace #file:.github/copilot/agents/09-security-engineer.md
#file:src/presentation/http/[controller].py
#file:src/application/use_cases/[use_case].py

Faça uma revisão de segurança focada em:
- Controle de acesso (IDOR, autorização)
- Validação de input
- Dados sensíveis em logs
```

---

## Definition of Done (DoD) por Story

Antes de abrir o PR, verifique:

- [ ] Implementação completa (todas as 4 camadas)
- [ ] Lint e formatter sem erros
- [ ] Testes unitários de domínio: cobertura 100% das regras de negócio
- [ ] Testes unitários de use case: happy path + todos os error cases
- [ ] Testes de integração: repositório com banco real
- [ ] Security review nos endpoints críticos
- [ ] Sem `TODO` ou `FIXME` não acompanhados de issue
- [ ] ADRs atualizados se houve decisão arquitetural
- [ ] Documentação de endpoint atualizada (OpenAPI)

---

## Abertura de PR

```bash
# Título: [tipo]([contexto]): [descrição curta]
# Ex: feat(orders): add process order use case

git push origin feature/[nome-da-feature]
gh pr create --title "feat(orders): add process order use case" \
  --body "Closes #[issue]\n\nImplements...\n\nChecklist:\n- [x] tests\n- [x] docs"
```

---

## Próxima Fase

→ [Fase 4: Quality Gate](04-quality-gate.md) após abertura do PR.
