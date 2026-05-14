# Fase 1: Inception

## Objetivo

Transformar uma **ideia ou problema de negócio** em um Discovery Report estruturado, pronto para entrar em design técnico.

**Duração típica**: 1-3 dias  
**Agente principal**: `01-project-discovery.md`  
**Participantes**: Product Owner, Tech Lead, Domain Expert

---

## Entrada

- Ideia / problema de negócio em linguagem natural
- Stakeholders disponíveis para entrevistas

---

## Como Ativar

```
@workspace #file:.github/copilot/sdlc/phases/01-inception.md
           #file:.github/copilot/agents/01-project-discovery.md

Quero criar [descrição do projeto/feature em 1-2 parágrafos].
```

---

## Atividades da Fase

### 1.1 — Product Vision Canvas

```markdown
## Product Vision Board

| Campo | Resposta |
|-------|---------|
| **Nome do produto/feature** | |
| **Problema que resolve** | |
| **Para quem** (persona principal) | |
| **Por que agora** (urgência) | |
| **Diferencial** (por que não fazer de outra forma) | |
| **Métrica de sucesso** (como saberemos que funcionou) | |
```

### 1.2 — Discovery de Requisitos

Execute as 5 seções do Agent 01 em sequência:

```
Seção 1 — Visão: Nome, problema, usuários, valor
Seção 2 — Domínio: Fluxos, eventos, regras, glossário
Seção 3 — NFRs: Volume, disponibilidade, segurança, stack
Seção 4 — Contexto estratégico: MVP, riscos, integrações
Seção 5 — Priorização: Must have / Should have / Nice to have
```

### 1.3 — Event Storming Inicial

Liste os eventos de negócio em ordem cronológica (post-its laranja):

```
[Evento 1 aconteceu] → [Evento 2 aconteceu] → [Evento 3 aconteceu]
        ↑                       ↑                       ↑
   [Comando]              [Regra/Policy]            [Ator/Sistema]
```

### 1.4 — Identificação de Bounded Contexts Candidatos

```
Context Map preliminar:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  [Contexto A]   │────▶│  [Contexto B]   │────▶│  [Contexto C]   │
│  Pedidos        │     │  Pagamentos      │     │  Notificações   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     (Core)                  (Generic)               (Supporting)
```

---

## Saídas (Artefatos)

### Discovery Report

```markdown
# Discovery Report — [Nome do Projeto]

**Data**: [data]
**Versão**: 1.0
**Autor**: [nome]
**Aprovado por**: [PO + Tech Lead]

## 1. Visão
**Sistema**: [nome]
**Problema**: [descrição clara do problema]
**Usuários**: [persona primária e secundárias]
**Proposta de valor**: [o que muda para o usuário]

## 2. Fluxos Principais
### Fluxo Feliz: [nome]
1. [Ator] [ação]
2. Sistema [reação]
3. [Resultado]

### Fluxo de Exceção: [nome]
...

## 3. Regras de Negócio
- RN-001: [regra]
- RN-002: [regra]

## 4. Glossário (Ubiquitous Language)
| Termo | Definição no Contexto |
|-------|----------------------|
| [termo] | [definição precisa] |

## 5. Requisitos Não-Funcionais
- **Volume**: [estimativa de requests/dia, usuários]
- **Disponibilidade**: [SLA esperado]
- **Latência**: [p95 aceitável]
- **Segurança**: [nível de sensibilidade dos dados]

## 6. Bounded Contexts Identificados
- **[Contexto A]** (Core): [responsabilidade]
- **[Contexto B]** (Supporting): [responsabilidade]

## 7. Integrações Externas
- [Sistema externo]: [tipo de integração, protocolo]

## 8. MVP Scope
### Must Have (Sprint 1)
- [ ] [feature crítica]

### Should Have (Sprint 2)
- [ ] [feature importante]

### Nice to Have (futuro)
- [ ] [feature desejável]

## 9. Riscos Identificados
| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| [risco] | Alta/Média/Baixa | Alto/Médio/Baixo | [estratégia] |
```

---

## Gate: Definition of Ready (DoR)

Marque todos os itens antes de avançar para a Fase 2:

- [ ] Discovery Report aprovado por PO e Tech Lead
- [ ] Glossário com termos do domínio definidos
- [ ] MVP scope delimitado e acordado
- [ ] Bounded Contexts identificados (pelo menos esboço)
- [ ] NFRs de volume e disponibilidade definidos
- [ ] Integrações externas mapeadas
- [ ] Riscos principais identificados com mitigação

**Sem o DoR completo, não inicie a Fase 2.**

---

## Próxima Fase

→ [Fase 2: Design](02-design.md) com o Discovery Report em mãos.
