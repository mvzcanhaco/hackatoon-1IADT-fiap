# Prompt: Bootstrap do Projeto (Novo Projeto)

## Como Usar

```
@workspace #file:.github/copilot/prompts/new-project/04-project-bootstrap.md
#file:.github/copilot/templates/hexagonal/domain/entity.md
#file:.github/copilot/templates/hexagonal/application/use-case.md

[cole o Architecture Design do passo anterior]

Gere o scaffold completo do projeto com código inicial para cada camada.
```

---

## Prompt Completo

Com base no Architecture Design fornecido, gere o **scaffold completo** do projeto com:
1. Estrutura de diretórios criada
2. Arquivos `__init__.py` (Python) ou `index.ts` (TypeScript)
3. Código inicial para cada building block identificado
4. Testes unitários base
5. Configuração do projeto

---

### O que Gerar

#### 1. Estrutura de Diretórios

Gere o script de criação:

```bash
#!/bin/bash
# bootstrap.sh — cria a estrutura do projeto

PROJECT_NAME="[nome-do-projeto]"
mkdir -p $PROJECT_NAME/{src,tests,docs}

# Domain
mkdir -p $PROJECT_NAME/src/[context]/{domain/{entities,value_objects,repositories,services,events,factories,exceptions},application/{use_cases,dto,ports},infrastructure/{repositories,adapters,config},presentation/{http,cli}}

# Tests
mkdir -p $PROJECT_NAME/tests/{unit/[context]/{domain,application},integration/[context]/infrastructure,e2e}

# Docs
mkdir -p $PROJECT_NAME/docs/{architecture/decisions,diagrams}

echo "✅ Estrutura criada com sucesso!"
```

---

#### 2. Entidades de Domínio

Para cada entidade identificada, gere o código completo:

```python
# src/[context]/domain/entities/[entity_name].py
```

Use o template em: `.github/copilot/templates/hexagonal/domain/entity.md`

---

#### 3. Value Objects

Para cada VO identificado:

```python
# src/[context]/domain/value_objects/[vo_name].py
```

Use o template em: `.github/copilot/templates/hexagonal/domain/value-object.md`

---

#### 4. Repository Interfaces

Para cada aggregate root:

```python
# src/[context]/domain/repositories/[entity]_repository.py
```

Use o template em: `.github/copilot/templates/hexagonal/domain/repository-interface.md`

---

#### 5. Use Cases com DTOs

Para cada use case:

```python
# src/[context]/application/use_cases/[use_case_name].py
```

Use o template em: `.github/copilot/templates/hexagonal/application/use-case.md`

---

#### 6. Exceções de Domínio

```python
# src/[context]/domain/exceptions/[context]_exceptions.py

class [Context]DomainError(Exception):
    """Base para todas as exceções de domínio do context [Context]."""
    pass

class [EntityName]NotFoundError([Context]DomainError):
    def __init__(self, entity_id: str) -> None:
        super().__init__(f"[EntityName] not found: {entity_id}")
        self.entity_id = entity_id

class Invalid[Field]Error([Context]DomainError):
    def __init__(self, value: str, reason: str) -> None:
        super().__init__(f"Invalid [field]: '{value}'. {reason}")
        self.value = value
```

---

#### 7. Configurações do Projeto

**`pyproject.toml` / `package.json`**:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src --cov-report=term-missing --cov-fail-under=80"

[tool.ruff]
line-length = 100
select = ["E", "W", "F", "I"]

[tool.mypy]
strict = true
ignore_missing_imports = true
```

**`.env.example`**:
```bash
# Application
APP_NAME=[nome-do-projeto]
APP_ENV=development
DEBUG=true

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/[db_name]

# External Services
[SERVICO_EXTERNO]_API_KEY=your_api_key_here
[SERVICO_EXTERNO]_BASE_URL=https://api.example.com
```

**`Makefile`**:
```makefile
.PHONY: install test lint format run

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

run:
	python -m src.main

coverage:
	pytest --cov=src --cov-report=html
	open htmlcov/index.html
```

---

#### 8. GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint
        run: ruff check src/ tests/

      - name: Type check
        run: mypy src/

      - name: Test
        run: pytest tests/ -v --cov=src --cov-fail-under=80

      - name: Architecture check
        run: |
          # Verifica que domínio não importa de infra
          python -c "
          import ast, sys, pathlib
          for f in pathlib.Path('src').rglob('*.py'):
              if '/domain/' in str(f):
                  tree = ast.parse(f.read_text())
                  for node in ast.walk(tree):
                      if isinstance(node, (ast.Import, ast.ImportFrom)):
                          imp = ast.dump(node)
                          if 'infrastructure' in imp or 'presentation' in imp:
                              print(f'VIOLATION: {f} imports from forbidden layer')
                              sys.exit(1)
          print('Architecture check passed')
          "
```

---

#### 9. `conftest.py` Base

```python
# tests/conftest.py
import pytest
from unittest.mock import Mock
# Import das interfaces do domínio para criar mocks tipados
```

---

## Checklist de Bootstrap

Após gerar o scaffold, verifique:

- [ ] Todos os diretórios criados
- [ ] `__init__.py` em cada diretório Python
- [ ] Entidades com comportamento (não anêmicas)
- [ ] Value Objects imutáveis (`frozen=True`)
- [ ] Interfaces (ports) definidas no domínio/aplicação
- [ ] Use cases com único método `execute()`
- [ ] Exceções de domínio customizadas
- [ ] Testes unitários base criados
- [ ] CI/CD configurado
- [ ] `.env.example` com todas as variáveis necessárias
- [ ] `Makefile` com comandos úteis
- [ ] `README.md` com instruções de setup

---

## Após o Bootstrap

Continue o desenvolvimento usando os agentes específicos:

```bash
# Para implementar um use case:
@workspace #file:.github/copilot/agents/04-developer.md

# Para modelar novas entidades:
@workspace #file:.github/copilot/agents/03-domain-modeler.md

# Para gerar testes:
@workspace #file:.github/copilot/agents/06-test-engineer.md

# Para revisar código:
@workspace #file:.github/copilot/agents/05-code-reviewer.md
```
