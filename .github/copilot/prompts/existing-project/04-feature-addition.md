# Prompt: Adição de Feature (Projeto Existente)

## Como Usar

```
@workspace #file:.github/copilot/prompts/existing-project/04-feature-addition.md
#file:.github/copilot/agents/04-developer.md
#file:src/domain/interfaces/[interface_relacionada].py
#file:src/application/use_cases/[use_case_relacionado].py

Feature: [descreva a nova feature]
Contexto: [descreva como ela se encaixa no sistema atual]
```

---

## Prompt Completo

Você vai adicionar uma **nova feature** a um projeto existente. Antes de escrever qualquer linha, faça o checklist completo de entendimento do contexto.

---

### Passo 1 — Entendimento do Contexto

Responda cada item antes de implementar:

```
1. QUAL é a feature exatamente?
   - Comportamento esperado em uma frase
   - Input: o que recebe
   - Output: o que retorna/faz

2. ONDE ela se encaixa na arquitetura?
   - Em qual bounded context?
   - Em qual camada começa? (nova entidade? novo use case? nova rota?)
   - Quais arquivos existentes são afetados?

3. QUAIS regras de negócio ela introduz?
   - Liste todas as validações necessárias
   - Liste os casos de erro
   - Liste as invariantes que devem ser mantidas

4. COMO ela interage com o código existente?
   - Reutiliza interfaces existentes? Quais?
   - Precisa de nova interface?
   - Modifica comportamento existente? (risco!)
   - É puramente aditiva? (seguro)

5. QUAIS testes são necessários?
   - Testes unitários do novo comportamento de domínio
   - Testes do use case (com mocks)
   - Testes de integração (se há novo adapter/repositório)
```

---

### Passo 2 — Impact Analysis

Antes de modificar qualquer arquivo existente:

```
PARA CADA ARQUIVO EXISTENTE A MODIFICAR:

1. Leia o arquivo completo
2. Identifique o que vai mudar
3. Verifique se há testes existentes para o que vai mudar
4. Se não há testes → escreva-os ANTES de modificar
5. Só então faça a modificação

REGRAS:
- Prefira ADICIONAR código a MODIFICAR código existente (Open/Closed Principle)
- Se precisar modificar → certifique que há testes cobrindo o comportamento atual
- Nunca quebre uma interface existente (cria nova, depreca a antiga)
```

---

### Passo 3 — Plano de Implementação

Gere um plano com a ordem correta:

```
Ordem recomendada de implementação (inside-out):

1. Domain (de dentro para fora)
   a. Novos Value Objects (se necessário)
   b. Novos métodos nas entidades existentes
   c. Novas entidades (se necessário)
   d. Nova interface/port (se necessário)
   e. Novos eventos de domínio

2. Application
   a. Novo DTO (Request + Response)
   b. Novo Use Case (ou extensão do existente)

3. Infrastructure (apenas se necessário)
   a. Novo método no repositório existente
   b. Novo adapter

4. Presentation
   a. Novo endpoint/rota/comando

5. Tests (escritos junto com cada camada)
   a. Testes de domínio → junto com 1.b / 1.c
   b. Testes de use case → junto com 2.b
   c. Testes de integração → junto com 3.a / 3.b
```

---

### Passo 4 — Implementação Guiada

Para cada arquivo a criar/modificar, siga o template:

#### Novo método em entidade existente

```
ANTES de modificar:
1. Leia: #file:src/domain/entities/[entity].py
2. Entenda o estado atual da entidade
3. Identifique qual invariante o novo método deve respeitar
4. Verifique os testes existentes: #file:tests/unit/domain/test_[entity].py

IMPLEMENTAÇÃO:
- Adicione o método respeitando as invariantes existentes
- Lance DomainException se invariantes forem violadas
- Retorne Domain Event se o estado mudar

TESTES (adicione ao arquivo de testes existente):
- Happy path do novo método
- Casos de erro (invariantes violadas)
- Interação com métodos existentes
```

#### Novo Use Case

```
TEMPLATE:
@dataclass
class [NovoCaso]Request:
    """DTO de entrada para [NovoCaso]."""
    [campo]: [tipo]  # [descrição]

@dataclass
class [NovoCaso]Response:
    """DTO de saída para [NovoCaso]."""
    [campo]: [tipo]

@dataclass
class [NovoCaso]UseCase:
    """
    Use Case: [NovoCaso]

    Responsabilidade: [uma frase]
    """
    [dependency]: [InterfaceType]

    def execute(self, request: [NovoCaso]Request) -> [NovoCaso]Response:
        ...
```

#### Novo Endpoint

```
TEMPLATE (FastAPI):
@router.[method]("/[path]", response_model=[ResponseSchema], status_code=[código])
async def [nome_da_operacao](
    payload: [RequestSchema],
    use_case: [NovoCaso]UseCase = Depends(get_[novo_caso]_use_case),
) -> [ResponseSchema]:
    """[Descrição do endpoint — aparece no Swagger]"""
    try:
        response = use_case.execute([NovoCaso]Request(...))
        return [ResponseSchema](...)
    except [DomainException] as e:
        raise HTTPException(status_code=422, detail=str(e))
```

---

### Checklist Final

Antes de considerar a feature pronta:

**Código**:
- [ ] Implementada inside-out (domínio → aplicação → infra → apresentação)
- [ ] Nenhuma lógica de negócio fora do domínio
- [ ] Dependências injetadas por construtor
- [ ] Nenhum arquivo existente quebrado

**Testes**:
- [ ] Testes unitários do novo comportamento de domínio
- [ ] Testes do use case com todos os cenários (happy path + erros)
- [ ] Testes de integração (se novo adapter/repositório)
- [ ] Todos os testes existentes continuam passando

**Qualidade**:
- [ ] Type hints completos
- [ ] Docstrings nas classes e métodos públicos
- [ ] Exceções de domínio específicas (não genéricas)
- [ ] Logging adequado

**Integração**:
- [ ] Endpoint documentado (Swagger/OpenAPI)
- [ ] Variáveis de ambiente adicionadas ao `.env.example`
- [ ] README atualizado (se necessário)

---

## Exemplo de Sessão Completa

```
# Para adicionar notificação por Slack ao sistema de detecção:

@workspace #file:.github/copilot/prompts/existing-project/04-feature-addition.md
#file:.github/copilot/agents/04-developer.md
#file:src/domain/interfaces/notification.py
#file:src/infrastructure/services/notification_services.py
#file:src/application/use_cases/process_video.py

Feature: Adicionar suporte a notificações via Slack
Contexto: O sistema já tem webhook e email. Precisa de Slack como novo canal.
Input: slack_webhook_url como parâmetro de notificação
Regra: Usar o mesmo padrão das notificações existentes
Sem breaking change: notification_type="slack" é novo, não muda os existentes
```
