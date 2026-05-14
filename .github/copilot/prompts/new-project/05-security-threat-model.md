# Prompt: Security Threat Model (Fase Inception/Design)

> Use este prompt na fase de design, após definir a arquitetura e antes de implementar.
> Referência: [.github/copilot/agents/09-security-engineer.md]

---

## Contexto Necessário

Antes de executar este prompt, inclua:

```
@workspace #file:.github/copilot/prompts/new-project/05-security-threat-model.md
           #file:.github/copilot/agents/09-security-engineer.md
           #file:.github/copilot/prompts/new-project/03-architecture-design.md
           #file:[architecture-document-or-domain-model]
```

---

## Prompt

```
Aja como Agent 09 (Security Engineer) e execute um threat model completo para o sistema descrito abaixo.

## Sistema

Nome: [NomeDoSistema]
Tipo: [API REST | SPA + API | Microservices | Mobile + API]
Dados sensíveis: [lista de dados PII, financeiros, médicos, etc.]
Usuários: [tipos de usuário e níveis de acesso]
Integrações externas: [serviços de terceiros, APIs externas, webhooks]

## Arquitetura (resumo)

[Cole aqui o Architecture Document do Passo 3 ou descreva os componentes principais]

---

## Entregáveis Esperados

### 1. Diagrama de Fluxo de Dados (DFD)
Liste todos os fluxos de dados entre:
  - Atores externos (usuários, sistemas externos)
  - Componentes internos (APIs, serviços, banco, cache, broker)
  - Trust boundaries (fronteiras de confiança)

### 2. Análise STRIDE por Componente
Para cada componente relevante (API gateway, serviço, banco, cache):
  - S — Spoofing: como um atacante pode se passar por outro?
  - T — Tampering: como os dados podem ser adulterados?
  - R — Repudiation: como negar uma ação realizada?
  - I — Information Disclosure: como dados podem vazar?
  - D — Denial of Service: como sobrecarregar o sistema?
  - E — Elevation of Privilege: como ganhar mais permissões?

### 3. Risk Register
Para cada ameaça identificada:
  | Ameaça | Componente | Impacto (1-5) | Probabilidade (1-5) | Score | Controle |
  |--------|-----------|---------------|---------------------|-------|---------|

### 4. Controles Recomendados por Camada

**Autenticação e Autorização:**
  - [ ] Estratégia de autenticação: [JWT | OAuth2 | API Key | Session]
  - [ ] Scopes e roles necessários
  - [ ] Política de expiração de tokens

**Input Validation:**
  - [ ] Validação no ponto de entrada (Schema/Zod/Pydantic)
  - [ ] Sanitização de dados antes de queries
  - [ ] Limites de tamanho e tipo

**Data Protection:**
  - [ ] Dados em trânsito: TLS 1.2+ obrigatório
  - [ ] Dados em repouso: campos sensíveis criptografados
  - [ ] Dados de log: PII mascarado (conforme LGPD)

**Infrastructure:**
  - [ ] Segredos via variável de ambiente ou secret manager
  - [ ] Princípio do menor privilégio nas credenciais de banco/serviços
  - [ ] Rate limiting por endpoint/IP/usuário

### 5. Checklist LGPD/GDPR
  - [ ] Mapeamento de dados pessoais coletados
  - [ ] Base legal para cada dado (consentimento, contrato, interesse legítimo)
  - [ ] Retenção e descarte de dados
  - [ ] Direitos do titular implementados (acesso, portabilidade, exclusão)
  - [ ] DPA com terceiros que processam dados pessoais

### 6. Security Acceptance Criteria (para DoD)
Liste os critérios que DEVEM estar presentes para considerar o sistema seguro:
  - [ ] [critério 1]
  - [ ] [critério 2]
  - [ ] [critério 3]

### 7. Vulnerabilidades OWASP Top 10 Relevantes
Identifique quais das 10 categorias são mais críticas para este sistema e descreva o controle específico:
  - A01 Broken Access Control: [controle específico]
  - A02 Cryptographic Failures: [controle específico]
  - A03 Injection: [controle específico]
  - [continue para as relevantes]

---

## Formato de Saída

Produza um documento Markdown estruturado com:
  1. Resumo executivo (3-5 linhas)
  2. DFD textual (lista de fluxos)
  3. Tabela STRIDE completa
  4. Risk Register priorizado
  5. Controles recomendados
  6. Checklist LGPD/GDPR
  7. Security Acceptance Criteria
  8. Top 3 ameaças críticas com plano de ação imediata
```

---

## Quando Usar Este Prompt

```
✅ Antes de implementar qualquer endpoint de autenticação
✅ Antes de integrar com sistemas de pagamento ou dados médicos
✅ Antes do primeiro deploy em produção
✅ Após mudança arquitetural significativa
✅ Durante revisão de segurança periódica (trimestral)

❌ Não substituir por análise superficial de checklist
❌ Não ignorar mesmo para MVPs (riscos básicos devem ser endereçados)
```

---

## Output Esperado

O resultado deve ser salvo como `docs/security/threat-model-[data].md` e linkado na PR que introduz a funcionalidade de autenticação/autorização.
