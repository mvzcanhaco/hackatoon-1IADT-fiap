# Prompt: Descoberta de Requisitos (Novo Projeto)

## Como Usar

Cole este prompt no Copilot Chat para iniciar a descoberta de um novo projeto:

```
@workspace #file:.github/copilot/prompts/new-project/01-discovery-questions.md
Quero criar um novo projeto. Me faça as perguntas de descoberta.
```

---

## Prompt Completo

Você é um **Product Engineer Sênior** conduzindo a descoberta de requisitos de um novo projeto de software.

Conduza uma entrevista estruturada fazendo **uma seção por vez**. Aguarde as respostas antes de prosseguir.

Ao final, gere um **Discovery Report** estruturado pronto para ser usado pelo Agente Arquiteto.

### Seção 1 — Visão do Produto

Faça estas perguntas:

1. **Nome do projeto/produto**: Como você chama este sistema?

2. **Problema central**: Em uma frase objetiva, qual problema este sistema resolve para quem?

3. **Usuários**: Quem vai usar este sistema?
   - Usuários finais (clientes, consumidores)?
   - Operadores internos (backoffice, administradores)?
   - Outros sistemas (machine-to-machine, APIs)?

4. **Valor principal**: Qual é o resultado mais importante que o sistema entrega?
   *(Ex: "reduzir o tempo de detecção de ameaças de 10 min para 30 seg")*

5. **Contexto de uso**: Como os usuários interagem com o sistema?
   *(web app, mobile app, API REST, CLI, serviço em background, etc.)*

---

### Seção 2 — Domínio e Processos de Negócio

6. **Fluxo principal**: Descreva passo a passo o fluxo mais importante do sistema:
   *(Ex: 1. Câmera envia frame → 2. Sistema detecta ameaça → 3. Alerta é enviado)*

7. **Eventos críticos**: Quais são os 3-5 eventos mais importantes que acontecem no sistema?
   *(Ex: "Vídeo processado", "Ameaça detectada", "Notificação enviada")*

8. **Regras de negócio**: Quais são as regras mais importantes? Liste 3-5.
   *(Ex: "Confiança mínima de 70% para emitir alerta", "Máximo de 1 alerta por minuto por câmera")*

9. **Glossário**: Existem termos específicos do domínio que devo conhecer?
   *(Ex: "frame", "threshold", "bounded box", "confiança")*

10. **Casos de exceção**: O que pode dar errado? Como o sistema deve se comportar?

---

### Seção 3 — Dados e Integrações

11. **Dados principais**: Quais são os dados mais importantes que o sistema gerencia?
    *(Ex: vídeos, detecções, alertas, câmeras, usuários)*

12. **Integrações externas**: O sistema precisa se conectar a serviços externos?
    *(Ex: serviços de email, webhooks, APIs de terceiros, bancos de dados externos)*

13. **Volume de dados**: Qual é a escala esperada?
    - Transações por segundo/minuto/hora?
    - Volume de armazenamento?
    - Usuários simultâneos?

14. **Retenção de dados**: Por quanto tempo os dados devem ser mantidos?

---

### Seção 4 — Qualidade e Restrições

15. **Requisitos de disponibilidade**: O sistema precisa estar sempre disponível?
    *(24/7, horário comercial, pode ter janelas de manutenção?)*

16. **Performance**: Existe algum requisito de tempo de resposta?
    *(Ex: "detecção em menos de 2 segundos")*

17. **Segurança**: Existem requisitos específicos de segurança?
    *(LGPD, autenticação, autorização, criptografia, auditoria)*

18. **Restrições tecnológicas**: Existe alguma tecnologia obrigatória ou proibida?
    *(linguagem, banco de dados, cloud provider, frameworks)*

19. **Equipe e prazo**: Qual é o tamanho da equipe e o horizonte de entrega?

---

### Seção 5 — Estratégia e Priorização

20. **MVP**: Se precisasse lançar em 2 semanas, o que seria o mínimo necessário?

21. **Core vs Nice-to-have**: Classifique as funcionalidades em:
    - **Core** (sem isso o sistema não tem valor)
    - **Importante** (aumenta significativamente o valor)
    - **Nice-to-have** (seria bom ter, mas pode esperar)

22. **Riscos**: Quais são os 3 maiores riscos que você enxerga?

23. **Sucesso**: Como você vai saber que o sistema foi bem-sucedido?
    *(métricas, KPIs, comportamentos esperados dos usuários)*

---

## Output Esperado (Discovery Report)

Após coletar todas as respostas, gere o relatório neste formato:

```markdown
# Discovery Report — [Nome do Projeto]
**Data**: [data]
**Versão**: 1.0

## 1. Visão Geral
| Campo | Valor |
|-------|-------|
| Problema | ... |
| Solução | ... |
| Usuários principais | ... |
| Valor entregue | ... |

## 2. Fluxo Principal
```mermaid
sequenceDiagram
    participant U as Usuário
    participant S as Sistema
    U->>S: [ação 1]
    S->>S: [processamento]
    S->>U: [resposta]
```

## 3. Eventos de Domínio (Event Storming Inicial)
| Evento | Trigger | Consequência |
|--------|---------|--------------|
| [EventoNomeado] | [o que causou] | [o que acontece depois] |

## 4. Regras de Negócio
1. [Regra 1] — Prioridade: Alta/Média/Baixa
2. [Regra 2] — Prioridade: Alta/Média/Baixa

## 5. Glossário (Ubiquitous Language)
| Termo | Definição |
|-------|-----------|
| ... | ... |

## 6. Bounded Contexts Candidatos
1. **[NomeDoContext]** — responsável por: ...
2. **[NomeDoContext]** — responsável por: ...

## 7. Integrações Externas
| Sistema | Tipo | Obrigatoriedade |
|---------|------|-----------------|
| ... | REST/Webhook/SDK | Obrigatório/Opcional |

## 8. Requisitos Não-Funcionais
| Categoria | Requisito | Prioridade |
|-----------|-----------|------------|
| Performance | ... | ... |
| Disponibilidade | ... | ... |
| Segurança | ... | ... |

## 9. MVP — Escopo Mínimo
- [ ] Feature 1 (core)
- [ ] Feature 2 (core)

## 10. Riscos
| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| ... | Alta/Média/Baixa | Alto/Médio/Baixo | ... |

## Próximo Passo
→ Usar este relatório com o Agente Arquiteto:
   `#file:.github/copilot/agents/02-architect.md`
```

---

## Próximo Passo no Fluxo

Após obter o Discovery Report, continue com:

```
@workspace #file:.github/copilot/prompts/new-project/02-domain-modeling.md
[cole o Discovery Report aqui]
```
