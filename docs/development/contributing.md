# Guia de Desenvolvimento

## Ambiente de Desenvolvimento

### Setup Inicial

1. **IDE Recomendada**
   - VSCode com extensões:
     - Python
     - Pylance
     - GitLens
     - Python Test Explorer

2. **Configuração do Git**
   ```bash
   git config --global user.name "Seu Nome"
   git config --global user.email "seu@email.com"
   ```

3. **Pre-commit Hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Padrões de Código

### 1. Estilo
- PEP 8
- Máximo 88 caracteres por linha
- Docstrings em todas as funções/classes
- Type hints obrigatórios

Exemplo:
```python
def process_frame(
    frame: np.ndarray,
    confidence: float = 0.5
) -> List[Detection]:
    """Processa um frame para detecção de objetos.

    Args:
        frame: Array numpy do frame
        confidence: Limiar de confiança

    Returns:
        Lista de detecções encontradas
    """
    pass
```

### 2. Estrutura de Arquivos
```
src/
├── domain/
│   ├── entities/
│   │   └── detection.py
│   └── interfaces/
│       └── detector.py
├── application/
│   └── use_cases/
│       └── process_video.py
└── infrastructure/
    └── services/
        └── weapon_detector.py
```

### 3. Testes
- pytest para testes unitários
- pytest-cov para cobertura
- Mocking para dependências externas

Exemplo:
```python
def test_process_frame():
    detector = WeaponDetector()
    frame = np.zeros((640, 480, 3))
    result = detector.process_frame(frame)
    assert len(result) >= 0
```

## Fluxo de Trabalho

### 1. Branches
- `main`: Produção
- `develop`: Desenvolvimento
- `feature/*`: Novas funcionalidades
- `fix/*`: Correções
- `release/*`: Preparação de release

### 2. Commits
```
feat: Adiciona detecção em tempo real
^--^  ^------------------------^
|     |
|     +-> Descrição no presente
|
+-------> Tipo: feat, fix, docs, style, refactor
```

### 3. Pull Requests
- Template obrigatório
- Code review necessário
- CI deve passar
- Squash merge preferido

## CI/CD

### GitHub Actions
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest
```

### Deploy
1. Staging
   ```bash
   ./deploy.sh staging
   ```

2. Produção
   ```bash
   ./deploy.sh production
   ```

## Debugging

### 1. Logs
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processando frame %d", frame_number)
```

### 2. Profiling
```python
import cProfile

def profile_detection():
    profiler = cProfile.Profile()
    profiler.enable()
    # código
    profiler.disable()
    profiler.print_stats()
```

### 3. GPU Monitoring
```python
import torch

def check_gpu():
    print(torch.cuda.memory_summary())
```

## Otimizações

### 1. GPU
- Batch processing
- Memória pinned
- Async data loading

### 2. CPU
- Multiprocessing
- NumPy vectorization
- Cache de resultados

## Segurança

### 1. Dependências
- Safety check
- Dependabot
- SAST scanning

### 2. Código
- Input validation
- Error handling
- Secrets management

## Documentação

### 1. Docstrings
```python
def detect_objects(
    self,
    frame: np.ndarray
) -> List[Detection]:
    """Detecta objetos em um frame.

    Args:
        frame: Frame no formato BGR

    Returns:
        Lista de detecções

    Raises:
        ValueError: Se o frame for inválido
    """
    pass
```

### 2. Sphinx
```bash
cd docs
make html
```

### 3. README
- Badges atualizados
- Exemplos práticos
- Troubleshooting comum 