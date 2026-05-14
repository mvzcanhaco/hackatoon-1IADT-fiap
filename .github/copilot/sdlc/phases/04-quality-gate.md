# Fase 4: Quality Gate

## Objetivo

Garantir que o código está pronto para produção antes do merge: revisão arquitetural, cobertura de testes, segurança e DoD verificado.

**Duração típica**: 2-4 horas  
**Agentes**: `05-code-reviewer`, `06-test-engineer`, `09-security-engineer`  
**Participantes**: Revisor(es), QA Engineer

---

## Entrada

- PR aberto com feature completa (Fase 3)
- CI verde (lint, format, testes passando)

---

## Como Ativar

```
@workspace #file:.github/copilot/sdlc/phases/04-quality-gate.md
           #file:.github/copilot/agents/05-code-reviewer.md
           #file:[arquivos do PR]

Revise como um quality gate pré-merge.
Contexto: [descreva o que a feature implementa]
PR: #[número]
```

---

## Checklist do Quality Gate

### ✅ Gate 1 — Automatizado (CI)

O CI deve passar **obrigatoriamente** antes da revisão manual:

```
□ Lint: zero warnings ou errors
□ Format: código formatado conforme padrão
□ Type check: mypy/tsc/clippy sem erros
□ Testes: zero falhas
□ Cobertura: ≥ 80% no código novo (relatório no CI)
□ SAST: Bandit/SemGrep/CodeQL sem findings críticos
```

### ✅ Gate 2 — Code Review (Agent 05)

```
@workspace #file:.github/copilot/agents/05-code-reviewer.md
#file:[todos os arquivos modificados no PR]

Revise este PR como quality gate.
Contexto: [feature]
Prioridade: verificar camadas 1 (Arquitetural) e 5 (Segurança) primeiro.
```

**Critérios de bloqueio**:
- 🔴 Qualquer finding Crítico → PR bloqueado
- 🟠 Findings Altos → devem ser corrigidos neste PR
- 🟡 Findings Médios → podem ser issues (technical debt)

### ✅ Gate 3 — Teste Engineer Review (Agent 06)

```
@workspace #file:.github/copilot/agents/06-test-engineer.md
#file:src/domain/entities/[entity].py
#file:src/application/use_cases/[use_case].py
#file:tests/[test_files]

Verifique se os testes cobrem todos os cenários obrigatórios:
- Happy path
- Not Found
- Invalid State
- Domain violation
- Persistence error
```

**Checklist de testes**:
- [ ] Cenário: Happy Path ✓
- [ ] Cenário: Not Found ✓
- [ ] Cenário: Already Exists (se aplicável) ✓
- [ ] Cenário: Invalid State ✓
- [ ] Cenário: Domain Rule Violation ✓
- [ ] Cenário: Persistence Error ✓
- [ ] Testes E2E para fluxos críticos ✓

### ✅ Gate 4 — Security Review (Agent 09)

Obrigatório para endpoints que:
- Processam dados pessoais (CPF, email, endereço)
- Realizam operações financeiras
- Expõem dados de outros usuários
- Recebem uploads de arquivos

```
@workspace #file:.github/copilot/agents/09-security-engineer.md
#file:src/presentation/http/[controller].py

Security review pré-merge.
Dados sensíveis: [lista]
```

**Critérios de bloqueio de segurança**:
- Vulnerabilidade OWASP crítica → bloqueio imediato
- Dado sensível em log → bloqueio imediato
- IDOR (acesso a dados de outro usuário) → bloqueio imediato

---

## Processo de Resolução de Findings

```
Finding identificado
      │
      ▼
 🔴 Crítico? ──── SIM ──→ Dev corrige agora
      │                         │
     NÃO                        ▼
      │               Re-revisão (Agent 05)
      ▼                         │
 🟠 Alto? ────── SIM ──→ Dev corrige neste PR
      │                         │
     NÃO                        ▼
      │               Re-revisão confirma correção
      ▼
 🟡 Médio? ──── SIM ──→ Criar issue para próximo sprint
      │
     NÃO
      │
      ▼
   Aprovado ✅
```

---

## Template de Aprovação de PR

Quando todos os gates passam, o revisor posta no PR:

```markdown
## ✅ Quality Gate — Aprovado

**Code Review Score**: [N]/10
**Cobertura**: [N]%
**Security**: Aprovado / N/A

### Resumo
[O que foi implementado e está correto]

### Observações (não bloqueadores)
- [item para acompanhamento]

**Status**: Aprovado para merge → main
```

---

## Template de Bloqueio de PR

```markdown
## ❌ Quality Gate — Bloqueado

### Bloqueadores (devem ser corrigidos antes do merge)

| # | Tipo | Severidade | Arquivo | Descrição |
|---|------|-----------|---------|-----------|
| 1 | Arquitetural | 🔴 | `src/...` | [problema] |
| 2 | Segurança | 🔴 | `src/...` | [vulnerabilidade] |

### Não-bloqueadores (issues criadas)
- [ ] #[número] — [descrição]

**Próximo passo**: corrija os bloqueadores e solicite re-revisão.
```

---

## Gate Final: DoD Verificado

- [ ] Todos os gates acima passam
- [ ] CI verde com cobertura ≥ 80%
- [ ] Zero findings críticos ou altos
- [ ] PR aprovado por ≥ 1 revisor
- [ ] Migrations testadas em ambiente local
- [ ] Documentação de API atualizada

---

## Próxima Fase

→ [Fase 5: Release](05-release.md) após merge aprovado na main.
