# Guia de Desenvolvimento

## Ambiente de Desenvolvimento

### Configuração do Ambiente

1. **Preparação do Sistema**

    - Instale Python 3.10+
    - Configure CUDA 11.8+ (para GPU)
    - Instale Git

2. **Configuração do Projeto**

    ```bash
    # Clone o repositório
    git clone https://github.com/seu-usuario/hackatoon-1iadt.git
    cd hackatoon-1iadt

    # Crie e ative o ambiente virtual
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    .\venv\Scripts\activate   # Windows

    # Instale as dependências
    pip install -r requirements.txt
    ```

### Ferramentas Recomendadas

- **IDE**: VSCode com extensões:
  - Python
  - Pylance
  - GitLens
  - Python Test Explorer

- **Linters e Formatadores**:
  - Ruff
  - Black
  - isort
  - mypy

## Padrões de Código

### Estilo de Código

1. **PEP 8**

    - Máximo 88 caracteres por linha
    - 4 espaços para indentação
    - Sem tabs, apenas espaços

2. **Docstrings**

    ```python
    def process_video(video_path: str, fps: int = 2) -> dict:
        """Processa um vídeo para detecção de objetos.

        Args:
            video_path: Caminho do arquivo de vídeo
            fps: Frames por segundo para processamento

        Returns:
            Dicionário com resultados da detecção

        Raises:
            FileNotFoundError: Se o vídeo não for encontrado
        """
        pass
    ```

3. **Type Hints**

    ```python
    from typing import List, Dict, Optional

    def detect_objects(
        frame: np.ndarray,
        confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        pass
    ```

## Testes

### Estrutura de Testes

```plaintext
tests/
├── unit/
│   ├── test_detector.py
│   └── test_video_processor.py
├── integration/
│   └── test_api.py
└── conftest.py
```

### Exemplos de Testes

```python
def test_detector_initialization():
    """Testa a inicialização do detector."""
    detector = WeaponDetector()
    assert detector.is_initialized()
    assert detector.device == "cuda" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_detection_threshold(threshold):
    """Testa diferentes limiares de detecção."""
    detector = WeaponDetector()
    result = detector.detect(sample_image, threshold=threshold)
    assert all(d["confidence"] >= threshold for d in result)
```

## Fluxo de Trabalho

### Git Flow

1. **Branches Principais**

    - `main`: Produção
    - `develop`: Desenvolvimento
    - `feature/*`: Novas funcionalidades
    - `fix/*`: Correções
    - `release/*`: Preparação de release

2. **Commits**

    ```plaintext
    feat: Adiciona detecção em tempo real
    ^--^  ^------------------------^
    |     |
    |     +-> Descrição no presente
    |
    +-------> Tipo: feat, fix, docs, style, refactor
    ```

### CI/CD

1. **GitHub Actions**

    ```yaml
    name: CI
    on: [push, pull_request]
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v2
          - name: Test
            run: pytest
    ```

2. **Deploy**

    ```bash
    # Deploy para staging
    ./deploy.sh staging

    # Deploy para produção
    ./deploy.sh production
    ```

## Debugging

### Logs

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processando frame %d", frame_number)
```

### Profiling

```python
import cProfile

def profile_detection():
    profiler = cProfile.Profile()
    profiler.enable()
    # código
    profiler.disable()
    profiler.print_stats()
```

## Otimizações

### GPU

- Batch processing
- Memória pinned
- Async data loading
- Cache de modelos

### CPU

- Multiprocessing
- NumPy vectorization
- Cache de resultados
- Otimização de memória
