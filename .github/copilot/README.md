# Copilot Dev Framework

Framework de desenvolvimento para GitHub Copilot com arquitetura hexagonal, DDD e boas práticas de engenharia de software.

---

## Como Usar

Este framework transforma o GitHub Copilot em um ecossistema completo de desenvolvimento. Ele funciona através de **referências de arquivo** no Copilot Chat — use `#file:` para incluir qualquer documento como contexto.

```
Copilot Chat: #file:.github/copilot/agents/02-architect.md
              Preciso desenhar a arquitetura do módulo de pagamentos
```

---

## Estrutura do Framework

```
.github/
├── copilot-instructions.md          ← Carregado automaticamente pelo Copilot
└── copilot/
    ├── README.md                    ← Este arquivo
    ├── agents/                      ← Personas especializadas
    │   ├── 01-project-discovery.md  ← Descobre requisitos (novo projeto)
    │   ├── 02-architect.md          ← Desenha a arquitetura
    │   ├── 03-domain-modeler.md     ← Modela o domínio
    │   ├── 04-developer.md          ← Gera código
    │   ├── 05-code-reviewer.md      ← Revisa código
    │   └── 06-test-engineer.md      ← Gera testes
    ├── skills/                      ← Base de conhecimento
    │   ├── hexagonal-architecture.md
    │   ├── domain-driven-design.md
    │   ├── design-patterns.md
    │   └── testing-pyramid.md
    ├── prompts/
    │   ├── new-project/             ← Fluxo para projetos novos
    │   │   ├── 01-discovery-questions.md
    │   │   ├── 02-domain-modeling.md
    │   │   ├── 03-architecture-design.md
    │   │   └── 04-project-bootstrap.md
    │   └── existing-project/        ← Fluxo para projetos existentes
    │       ├── 01-context-capture.md
    │       ├── 02-analysis-report.md
    │       ├── 03-modernization-plan.md
    │       └── 04-feature-addition.md
    └── templates/
        ├── hexagonal/               ← Templates de código por camada
        │   ├── domain/
        │   ├── application/
        │   ├── infrastructure/
        │   └── presentation/
        └── patterns/                ← Catálogo de Design Patterns
```

---

## Fluxos de Trabalho

### Fluxo 1: Novo Projeto

Execute na ordem indicada. Cada prompt usa o output do anterior como input.

```
Passo 1 → #file:.github/copilot/prompts/new-project/01-discovery-questions.md
Passo 2 → #file:.github/copilot/prompts/new-project/02-domain-modeling.md
Passo 3 → #file:.github/copilot/prompts/new-project/03-architecture-design.md
Passo 4 → #file:.github/copilot/prompts/new-project/04-project-bootstrap.md
```

### Fluxo 2: Projeto Existente

```
Passo 1 → #file:.github/copilot/prompts/existing-project/01-context-capture.md
Passo 2 → #file:.github/copilot/prompts/existing-project/02-analysis-report.md
Passo 3 → #file:.github/copilot/prompts/existing-project/03-modernization-plan.md
Passo 4 → #file:.github/copilot/prompts/existing-project/04-feature-addition.md
```

---

## Agentes — Quando Usar Cada Um

| Agente              | Use quando...                                              |
|---------------------|------------------------------------------------------------|
| `01-project-discovery` | Iniciar um projeto novo ou mapear requisitos           |
| `02-architect`      | Desenhar módulos, camadas, integrações                     |
| `03-domain-modeler` | Criar entidades, value objects, aggregates                 |
| `04-developer`      | Implementar use cases, repositories, adapters              |
| `05-code-reviewer`  | Revisar PRs, detectar violações arquiteturais              |
| `06-test-engineer`  | Criar suítes de teste, mocks, fixtures                     |

---

## Exemplo de Sessão Completa

### Iniciando um Novo Projeto

```
1. Abra o Copilot Chat no VS Code
2. Digite:
   @workspace #file:.github/copilot/agents/01-project-discovery.md
   Quero construir um sistema de gestão de biblioteca

3. O agente fará perguntas estruturadas sobre seu domínio

4. Após a descoberta, passe para o arquiteto:
   @workspace #file:.github/copilot/agents/02-architect.md
   [cole o output do passo anterior]
   Projete a arquitetura hexagonal para este sistema

5. Continue com modelagem de domínio:
   @workspace #file:.github/copilot/agents/03-domain-modeler.md
   [cole o output do passo anterior]
   Modele as entidades e agregados
```

### Trabalhando em Projeto Existente

```
1. Captura de contexto automática:
   @workspace #file:.github/copilot/prompts/existing-project/01-context-capture.md
   #file:src/domain/entities/detection.py
   #file:src/application/use_cases/process_video.py
   #file:src/infrastructure/services/weapon_detector.py
   Analise o contexto deste projeto

2. O agente gera um relatório completo do estado atual

3. Com o relatório, planeje melhorias ou novas features
```

---

## Skills como Referências

As skills são documentos de conhecimento. Referencie-as quando precisar de orientação específica:

```bash
# Quando tiver dúvidas sobre onde colocar algo:
#file:.github/copilot/skills/hexagonal-architecture.md
Onde devo colocar a lógica de cache?

# Quando for modelar domínio:
#file:.github/copilot/skills/domain-driven-design.md
Como modelar o aggregate de Pedido com seus itens?

# Quando escolher um padrão:
#file:.github/copilot/skills/design-patterns.md
Qual pattern usar para suportar múltiplos métodos de pagamento?
```

---

## Instalação em Outro Projeto

Copie a pasta `.github/copilot/` e o arquivo `.github/copilot-instructions.md` para o root do seu projeto. O Copilot carregará as instruções automaticamente.

```bash
# Via curl (substitua pela URL do seu repositório)
curl -sL <URL_DO_SCRIPT_DE_INSTALACAO> | bash

# Via cópia manual
cp -r .github/copilot /seu-projeto/.github/copilot
cp .github/copilot-instructions.md /seu-projeto/.github/copilot-instructions.md
```
