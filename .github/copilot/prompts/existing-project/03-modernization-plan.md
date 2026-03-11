# Prompt: Plano de Modernização (Projeto Existente)

## Como Usar

```
@workspace #file:.github/copilot/prompts/existing-project/03-modernization-plan.md
#file:.github/copilot/skills/hexagonal-architecture.md
#file:.github/copilot/skills/domain-driven-design.md

[cole o Analysis Report]

Crie o plano de modernização priorizando [arquitetura / qualidade / performance / testes].
Restrição: [sem breaking changes / apenas backend / prazo de X sprints]
```

---

## Prompt Completo

Com base no Analysis Report, crie um **plano de modernização incremental e seguro** que evolua o código sem quebrar o que já funciona.

**Princípio**: *"Se mexe, não quebra. Refatora com testes."*

---

### Estratégia de Modernização

#### Fase 0 — Safety Net (Antes de qualquer mudança)

```
OBJETIVO: Criar testes suficientes para refatorar com segurança.

Para cada módulo a ser modificado:
1. Identificar comportamentos existentes (documentados ou não)
2. Escrever testes de caracterização (Characterization Tests)
   - Testam o que o código FAZ atualmente, não o que DEVERIA fazer
   - São o safety net para refatorações
3. Atingir cobertura mínima de 70% antes de refatorar

Characterization Test Example:
def test_[modulo]_behaves_as_currently():
    """Este teste documenta o comportamento atual do módulo.
    Não necessariamente é o comportamento correto — é o atual."""
    result = [chame_o_codigo_existente]()
    assert result == [valor_atual]  # pode ser um bug documentado
```

---

#### Fase 1 — Extração de Domínio (Strangler Fig Pattern)

```
OBJETIVO: Isolar a lógica de negócio sem mudar o comportamento.

Padrão: Strangler Fig — envolver código legado aos poucos.

Passo 1: Identificar lógica de negócio no lugar errado (controller, service genérico)
Passo 2: Criar entidade/VO de domínio com essa lógica
Passo 3: Fazer o código existente chamar a nova entidade
Passo 4: Verificar que todos os testes passam
Passo 5: Remover a lógica duplicada do lugar original

# Antes da refatoração:
class VideoProcessor:
    def process(self, path: str, threshold: float) -> dict:
        if threshold < 0 or threshold > 1:  # regra no lugar errado
            raise ValueError("threshold must be between 0 and 1")
        ...

# Depois — Step 1: criar VO com a regra
@dataclass(frozen=True)
class DetectionThreshold:
    value: float
    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise InvalidThresholdError(self.value)

# Depois — Step 2: usar o VO
class VideoProcessor:
    def process(self, path: str, threshold: float) -> dict:
        threshold_vo = DetectionThreshold(threshold)  # regra encapsulada no VO
        ...
```

---

#### Fase 2 — Introdução de Ports & Adapters

```
OBJETIVO: Introduzir interfaces para desacoplar dependências externas.

Para cada dependência concreta no domínio/aplicação:
1. Criar interface (port) no domínio ou aplicação
2. Fazer a implementação existente implementar a interface
3. Injetar a interface no construtor (não a implementação)
4. Criar adapter para a implementação original

Antes:
class ProcessVideoUseCase:
    def __init__(self):
        self.detector = WeaponDetector()  # acoplado à implementação

Depois:
class ProcessVideoUseCase:
    def __init__(self, detector: DetectorInterface):  # depende da interface
        self._detector = detector
```

---

#### Fase 3 — Reorganização Estrutural

```
OBJETIVO: Mover arquivos para a estrutura hexagonal correta.

Regra: Mova um arquivo por vez, rodando os testes após cada movimentação.

Roteiro:
1. Criar estrutura nova (não apagar a antiga ainda)
2. Criar arquivo novo no lugar correto com o código refatorado
3. Fazer o arquivo antigo importar do novo (bridge temporário)
4. Atualizar todos os imports para apontar para o novo lugar
5. Remover o bridge e o arquivo antigo
6. Rodar todos os testes
```

---

### Template do Plano de Modernização

#### Sprint Planning View

```markdown
## Épico: Modernização de [Nome do Módulo]

### Sprint 1 — Safety Net
**Objetivo**: Cobertura de testes suficiente para refatorar com segurança

| Task | Arquivo | Estimativa | Prioridade |
|------|---------|------------|------------|
| Caracterização tests para [modulo] | tests/characterization/ | 4h | Alta |
| ... | | | |

**Definition of Done**:
- [ ] Cobertura >= 70% dos comportamentos críticos
- [ ] Todos os testes passando na CI

---

### Sprint 2 — Extração de Domínio
**Objetivo**: Mover regras de negócio para o lugar correto

| Task | De | Para | Estimativa |
|------|----|------|------------|
| Criar [Entity] com regras de [LocalAtual] | [arquivo]:L[N] | domain/entities/ | 3h |
| Criar [ValueObject] para [primitivo] | vários arquivos | domain/value_objects/ | 2h |
| ... | | | |

**Definition of Done**:
- [ ] Todos os testes existentes passando
- [ ] Novas entidades com testes unitários próprios
- [ ] Zero regras de negócio fora do domínio

---

### Sprint 3 — Desacoplamento de Infraestrutura
**Objetivo**: Introduzir interfaces (ports) para dependências externas

| Task | Interface | Implementação Atual | Estimativa |
|------|-----------|--------------------|----|
| Criar [InterfaceName]Port | domain/interfaces/ | infrastructure/services/[atual].py | 2h |
| Injetar por construtor em [UseCase] | application/ | — | 1h |
| ... | | | |

---

### Sprint 4 — Cobertura de Testes Final
**Objetivo**: Cobertura alvo atingida com testes de qualidade

| Camada | Cobertura Atual | Meta | Delta |
|--------|----------------|------|-------|
| domain | [%] | 95% | [+%] |
| application | [%] | 90% | [+%] |
| infrastructure | [%] | 80% | [+%] |
```

---

## Output Esperado

```markdown
# Plano de Modernização — [Projeto]

## Contexto
- **Estado atual**: [resumo do que foi encontrado]
- **Estado alvo**: [onde queremos chegar]
- **Estratégia**: Strangler Fig / Big Bang Rewrite / Incremental
- **Restrição**: [breaking changes ok / zero downtime / etc.]

## Cronograma Macro
| Sprint | Foco | Entregáveis | Risco |
|--------|------|-------------|-------|
| 1 | Safety Net | Testes de caracterização | Baixo |
| 2 | Domínio | Entidades e VOs extraídos | Médio |
| 3 | Desacoplamento | Interfaces introduzidas | Médio |
| 4 | Cobertura | Cobertura alvo atingida | Baixo |

## Backlog Detalhado
[lista de tasks priorizadas]

## Métricas de Acompanhamento
- Cobertura de testes: [atual] → [meta]
- Número de violações arquiteturais: [atual] → 0
- Complexidade média por método: [atual] → <5
- Tempo de execução dos testes: [atual] → <30s

## Riscos e Mitigações
[tabela de riscos]

## Próximo Passo
→ `#file:.github/copilot/agents/04-developer.md`
  Para implementar cada task do backlog
```
