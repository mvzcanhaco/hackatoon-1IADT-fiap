# Prompt: Captura de Contexto (Projeto Existente)

## Como Usar

```
@workspace #file:.github/copilot/prompts/existing-project/01-context-capture.md
#file:src/domain/entities/[entidade_principal].py
#file:src/application/use_cases/[use_case_principal].py
#file:src/infrastructure/services/[servico_principal].py
#file:[qualquer outro arquivo relevante]

Analise o contexto completo deste projeto e gere o relatório.
```

**Dica**: Inclua quantos `#file:` forem necessários para dar contexto completo.
Para projetos grandes, comece pelos arquivos mais representativos do domínio.

---

## Prompt Completo

Você é um **Tech Lead** fazendo um deep-dive em um projeto existente para entender completamente seu estado atual antes de qualquer modificação.

Analise todos os arquivos fornecidos e gere um relatório de contexto completo.

---

### O que Analisar

#### 1. Arquitetura Atual

```
Identifique:
- Qual arquitetura está sendo usada? (hexagonal, MVC, layered, monolito, etc.)
- Existe separação de camadas? É respeitada?
- Quais são os módulos/contextos existentes?
- Como as dependências fluem entre as camadas?
- Existem violações arquiteturais evidentes?
```

#### 2. Domínio de Negócio

```
Mapeie:
- Quais entidades existem? (classes com identidade)
- Quais são os value objects? (classes imutáveis por valor)
- Quais são as regras de negócio implementadas?
- Quais são os casos de uso do sistema?
- Qual é o vocabulário do domínio (termos usados no código)?
- Existem eventos de domínio?
```

#### 3. Qualidade do Código

```
Avalie:
- Aderência a SOLID
- Presença de code smells (God Classes, Long Methods, etc.)
- Cobertura de testes (se houver)
- Nomenclatura e clareza
- Documentação (docstrings, comentários)
- Tratamento de erros
```

#### 4. Débito Técnico

```
Identifique:
- Duplicação de código
- Abstrações faltando ou excessivas
- Dependências problemáticas
- Código morto
- TODO/FIXME/HACK pendentes
- Versões desatualizadas de dependências
```

#### 5. Integrações e Dependências

```
Mapeie:
- Dependências externas (bibliotecas, frameworks)
- Integrações com serviços externos (APIs, bancos, mensageria)
- Pontos de entrada (HTTP, CLI, eventos, workers)
- Pontos de saída (bancos, APIs, notificações)
```

---

### Inventário Completo

Para cada arquivo analisado, produza:

```
Arquivo: [caminho]
Tipo: [Entity / UseCase / Repository / Service / Controller / Config / Test / etc.]
Responsabilidade: [uma frase]
Dependências: [lista de imports relevantes]
Problemas identificados: [lista ou "nenhum"]
```

---

## Output Esperado

```markdown
# Context Capture Report — [Nome do Projeto]

**Data da análise**: [data]
**Arquivos analisados**: [N]
**Linhas de código**: [estimativa]

---

## 1. Visão Geral do Sistema

### O que o sistema faz
[descrição objetiva baseada no código]

### Stack Tecnológica
| Categoria | Tecnologia | Versão |
|-----------|-----------|--------|
| Linguagem | Python / TypeScript | ... |
| Framework | FastAPI / Django / Express | ... |
| Banco de dados | PostgreSQL / MongoDB | ... |
| IA/ML | [se aplicável] | ... |

---

## 2. Arquitetura Atual

### Estrutura Identificada
```
[reproduza a estrutura real de diretórios]
```

### Avaliação Arquitetural
| Aspecto | Situação | Observações |
|---------|----------|-------------|
| Separação de camadas | ✅ Presente / ⚠️ Parcial / ❌ Ausente | ... |
| Inversão de dependência | ✅ / ⚠️ / ❌ | ... |
| Interfaces/Abstrações | ✅ / ⚠️ / ❌ | ... |
| Testabilidade | ✅ / ⚠️ / ❌ | ... |

### Violações Encontradas
1. [descrição da violação] — arquivo: [caminho]:linha [N]
2. ...

---

## 3. Mapa do Domínio

### Entidades Identificadas
| Entidade | Arquivo | Atributos | Comportamentos | Observações |
|----------|---------|-----------|----------------|-------------|
| [Nome] | [path] | [lista] | [métodos] | [notas] |

### Value Objects
| VO | Arquivo | Imutável? | Observações |
|----|---------|-----------|-------------|
| [Nome] | [path] | ✅/❌ | ... |

### Use Cases / Serviços
| Use Case | Arquivo | Input | Output | Observações |
|----------|---------|-------|--------|-------------|
| [Nome] | [path] | ... | ... | ... |

### Regras de Negócio Identificadas
1. [regra] — implementada em: [arquivo:linha]
2. ...

---

## 4. Inventário de Arquivos

| Arquivo | Tipo | Responsabilidade | Problemas |
|---------|------|------------------|-----------|
| [path] | Entity | ... | Nenhum / [lista] |

---

## 5. Débito Técnico

### Alto Impacto (Urgente)
- [ ] [problema] — [arquivo] — [impacto]

### Médio Impacto (Planejado)
- [ ] [problema] — [arquivo] — [impacto]

### Baixo Impacto (Backlog)
- [ ] [problema] — [arquivo] — [impacto]

---

## 6. Cobertura de Testes

| Módulo | Testes Existentes | Cobertura Estimada |
|--------|------------------|-------------------|
| domain | ✅/❌ | [%] |
| application | ✅/❌ | [%] |
| infrastructure | ✅/❌ | [%] |

---

## 7. Oportunidades de Melhoria

### Quick Wins (baixo esforço, alto impacto)
1. ...

### Melhorias Estruturais (médio esforço)
1. ...

### Refatorações Estratégicas (alto esforço)
1. ...

---

## 8. Riscos ao Modificar

| Risco | Probabilidade | Área de Impacto | Mitigação |
|-------|---------------|-----------------|-----------|
| [risco] | Alta/Média/Baixa | [módulo] | [ação] |

---

## Próximo Passo
→ `#file:.github/copilot/prompts/existing-project/02-analysis-report.md`
  Para análise aprofundada de uma área específica

→ `#file:.github/copilot/prompts/existing-project/03-modernization-plan.md`
  Para plano de modernização completa

→ `#file:.github/copilot/prompts/existing-project/04-feature-addition.md`
  Para adicionar uma nova feature respeitando o código existente
```
