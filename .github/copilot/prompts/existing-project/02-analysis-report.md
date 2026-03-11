# Prompt: Relatório de Análise Aprofundada (Projeto Existente)

## Como Usar

```
@workspace #file:.github/copilot/prompts/existing-project/02-analysis-report.md
#file:.github/copilot/skills/hexagonal-architecture.md

[cole o Context Capture Report]

Faça análise aprofundada do módulo [nome do módulo].
Foco: [arquitetura / qualidade / performance / segurança / testes]
```

---

## Prompt Completo

Com base no Context Capture Report e nos arquivos do workspace, realize uma **análise aprofundada** do módulo ou aspecto especificado.

---

### Análise 1 — Aderência à Arquitetura Hexagonal

```
Para cada arquivo do módulo, verifique:

DOMÍNIO:
□ Entidades sem dependências de framework
□ Value Objects imutáveis com validação
□ Interfaces (ports) definidas corretamente
□ Regras de negócio nas entidades, não nos use cases ou controllers
□ Exceptions de domínio customizadas

APLICAÇÃO:
□ Use cases com único método execute()
□ DTOs sem lógica de negócio
□ Dependências injetadas via construtor
□ Nenhum acesso direto a repositório concreto

INFRAESTRUTURA:
□ Implementa interfaces do domínio
□ Mapeamento domínio ↔ persistência isolado
□ Nenhuma lógica de negócio
□ Tratamento adequado de erros de infraestrutura

APRESENTAÇÃO:
□ Apenas chama use cases
□ Converte erros de domínio em respostas HTTP adequadas
□ Não acessa repositórios ou serviços diretamente
□ Validação de input no ponto de entrada
```

---

### Análise 2 — Qualidade e Manutenibilidade

Para cada classe/função identificada:

```
MÉTRICAS DE COMPLEXIDADE:
- Número de linhas (>200 = God Class candidate)
- Número de métodos (>10 = muitas responsabilidades)
- Profundidade de herança (>3 = design smell)
- Número de dependências no construtor (>5 = muitas responsabilidades)
- Complexidade ciclomática (if/elif/for/while por método — >5 = complexo)

CODE SMELLS A IDENTIFICAR:
□ God Class — classe faz tudo
□ Long Method — método com mais de 20 linhas
□ Feature Envy — método usa mais dados de outra classe
□ Data Clumps — mesmos parâmetros sempre juntos
□ Primitive Obsession — usa primitivos onde VOs seriam melhores
□ Switch/if-elif chains — candidate para Strategy/Factory
□ Shotgun Surgery — mudança simples exige alterar muitos arquivos
□ Divergent Change — classe muda por múltiplas razões
□ Parallel Inheritance Hierarchies — criar subclasse X exige criar subclasse Y
```

---

### Análise 3 — Cobertura de Testes

```
Para cada módulo, analise:

TESTES EXISTENTES:
- Quais comportamentos têm cobertura?
- Quais comportamentos NÃO têm cobertura?
- Os testes testam comportamento ou implementação?
- Os mocks são usados adequadamente?
- Testes frágeis (dependem de ordem, estado global)?

LACUNAS CRÍTICAS:
- Regras de negócio sem teste
- Casos de erro não testados
- Integrações não testadas
- Edge cases ignorados
```

---

### Análise 4 — Segurança (quando aplicável)

```
CHECKLIST OWASP:
□ SQL Injection — queries parametrizadas?
□ XSS — outputs sanitizados?
□ IDOR — validação de ownership antes de retornar dados?
□ Dados sensíveis em logs?
□ Secrets hardcoded no código?
□ Validação de input nos pontos de entrada?
□ Rate limiting em operações custosas?
□ Autenticação e autorização adequadas?
□ Dependências com vulnerabilidades conhecidas?
```

---

### Análise 5 — Performance

```
PONTOS DE ATENÇÃO:
□ N+1 queries (loop que faz query no banco a cada iteração)
□ Queries sem índice em tabelas grandes
□ Carregamento de dados desnecessários (SELECT *)
□ Ausência de paginação em listas grandes
□ Processamento síncrono onde assíncrono seria melhor
□ Ausência de cache em operações custosas e repetitivas
□ Alocação desnecessária de memória em loops
```

---

## Output Esperado

```markdown
# Analysis Report — [Módulo/Aspecto] — [Projeto]

## Resumo Executivo
| Categoria | Score | Observação |
|-----------|-------|------------|
| Aderência Arquitetural | [0-10] | ... |
| Qualidade de Código | [0-10] | ... |
| Cobertura de Testes | [0-10] | ... |
| Segurança | [0-10] | ... |
| Performance | [0-10] | ... |

## Problemas Críticos (bloqueiam evolução segura)
### CRÍTICO 1: [Nome do Problema]
**Arquivo**: `[caminho]:linha [N]`
**Impacto**: [o que pode dar errado]
**Solução**: [como resolver]

## Problemas de Qualidade
[lista formatada com código atual e código sugerido]

## Lacunas de Teste
### Cenários sem cobertura:
1. [comportamento não testado] — [por que é importante testar]

## Recomendações Prioritizadas
| Prioridade | Ação | Esforço | Impacto |
|------------|------|---------|---------|
| 1 | [ação] | P (1-2h) / M (1-2d) / G (1-2s) | Alto/Médio/Baixo |

## Próximo Passo
→ `#file:.github/copilot/prompts/existing-project/03-modernization-plan.md`
```
