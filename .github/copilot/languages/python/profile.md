# Perfil de Linguagem: Python

> **Referências Canônicas**:
> - [PEP 8](https://peps.python.org/pep-0008/) — Style Guide for Python Code
> - [PEP 257](https://peps.python.org/pep-0257/) — Docstring Conventions
> - [PEP 484](https://peps.python.org/pep-0484/) / [526](https://peps.python.org/pep-0526/) / [604](https://peps.python.org/pep-0604/) — Type Hints
> - [PEP 20](https://peps.python.org/pep-0020/) — The Zen of Python
> - [Python Docs 3.12](https://docs.python.org/3.12/)

---

## Detecção Automática

Copilot ativa este perfil quando o workspace contém:
- Arquivos `.py`, `pyproject.toml`, `requirements.txt`, `setup.cfg`
- `python_requires` no `pyproject.toml`

---

## Versão Mínima Suportada

Verifique sempre em `pyproject.toml`:
```toml
[project]
requires-python = ">=3.11"  # leia este campo antes de usar features
```

**Nunca use** features de versão superior à declarada em `requires-python`.

---

## Convenções Obrigatórias (PEP 8)

### Nomenclatura

| Elemento | Convenção | Exemplo |
|----------|-----------|---------|
| Funções e variáveis | `snake_case` | `process_video()`, `frame_count` |
| Classes | `PascalCase` | `WeaponDetector`, `DetectionResult` |
| Constantes | `UPPER_SNAKE_CASE` | `MAX_FRAMES`, `DEFAULT_THRESHOLD` |
| Módulos e pacotes | `snake_case` | `weapon_detector.py`, `use_cases/` |
| Parâmetros privados | `_snake_case` | `_session`, `_cache` |
| Name mangling | `__snake_case` | `__internal_state` (use raramente) |
| Dunder methods | `__method__` | `__init__`, `__repr__`, `__eq__` |

### Formatação
```python
# Linhas: máximo 99 caracteres (configurar no ruff/black)
# Indentação: 4 espaços (nunca tabs)
# Imports: ordenados (stdlib → third-party → local), sem wildcards
import os                        # stdlib
import sys

import numpy as np               # third-party
from pydantic import BaseModel

from .entities.detection import Detection  # local (relativo)
from ..application.use_cases import ProcessVideoUseCase
```

---

## Type Hints Obrigatórios (PEP 484/526/604)

```python
# Python 3.10+ — use X | Y em vez de Union[X, Y]
def process(
    video_path: str,
    threshold: float,
    callback: Callable[[DetectionResult], None] | None = None,
) -> DetectionResult:
    ...

# Python 3.9+ — use list[X] em vez de List[X]
def get_detections(frame: VideoFrame) -> list[Detection]:
    ...

# Python 3.12+ — TypeVar com sintaxe nova
type Vector[T] = list[T]  # PEP 695

# dataclass com type hints completos
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class DetectionConfig:
    threshold: float = 0.7
    fps: int = 5
    max_frames: int = field(default=300)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)
    MAX_FPS: ClassVar[int] = 30  # ClassVar = não é campo de instância
```

---

## Docstrings (PEP 257 + Google Style)

```python
def process_video(
    video_path: str,
    config: DetectionConfig,
) -> DetectionResult:
    """Processa vídeo e retorna resultado de detecção.

    Analisa o vídeo frame a frame usando o modelo OWLV2,
    identificando objetos potencialmente perigosos.

    Args:
        video_path: Caminho absoluto para o arquivo de vídeo.
            Formatos suportados: MP4, AVI, MOV, MKV.
        config: Configurações de detecção incluindo threshold
            e FPS de amostragem.

    Returns:
        DetectionResult com lista de detecções, métricas de
        processamento e informações do dispositivo usado.

    Raises:
        VideoNotFoundError: Se o arquivo não existe ou não é legível.
        InvalidThresholdError: Se threshold não está entre 0 e 1.
        DetectorNotInitializedError: Se o modelo não foi carregado.

    Example:
        >>> config = DetectionConfig(threshold=0.8, fps=5)
        >>> result = process_video("/path/to/video.mp4", config)
        >>> print(f"Detectadas: {len(result.detections)} ameaças")
    """
```

---

## Tratamento de Erros

```python
# ✅ Hierarquia de exceções de domínio
class AppError(Exception):
    """Base para todas as exceções da aplicação."""

class DomainError(AppError):
    """Erros de regra de negócio — esperados, tratáveis."""

class InfrastructureError(AppError):
    """Erros de infra — inesperados, loguear e re-raise."""

class VideoNotFoundError(DomainError):
    def __init__(self, path: str) -> None:
        super().__init__(f"Video not found: {path!r}")
        self.path = path

# ✅ Uso correto
try:
    result = process_video(path, config)
except VideoNotFoundError as e:
    logger.warning("Video not found: %s", e.path)
    raise HTTPException(status_code=404, detail=str(e))
except Exception:
    logger.exception("Unexpected error processing video")  # inclui traceback
    raise

# ❌ Proibido
try:
    ...
except Exception:
    pass  # silencia todos os erros — NUNCA faça isso
except Exception as e:
    print(e)  # não usa logger — NUNCA em produção
```

---

## Logging (padrão stdlib)

```python
import logging

# Configuração do logger (por módulo, não por arquivo)
logger = logging.getLogger(__name__)  # usa o nome do módulo

# Uso correto (lazy formatting — mais eficiente)
logger.debug("Processing frame %d of %d", current, total)
logger.info("Detection completed: %d threats found", threat_count)
logger.warning("Low confidence detection: %.2f < %.2f", conf, threshold)
logger.error("Failed to load model: %s", model_path)
logger.exception("Unexpected error")  # inclui traceback automaticamente

# ❌ Proibido — interpolação eager (string formatada mesmo sem debug ativo)
logger.debug(f"Processing frame {current} of {total}")  # use % em vez de f-string
```

---

## Padrões Modernos Python 3.11+

```python
# ExceptionGroup (PEP 654) — Python 3.11+
try:
    ...
except* ValueError as eg:
    for e in eg.exceptions:
        logger.warning("Validation error: %s", e)

# Self type (PEP 673) — Python 3.11+
from typing import Self

@dataclass
class Builder:
    def with_config(self, config: Config) -> Self:
        self._config = config
        return self

# tomllib (PEP 680) — Python 3.11+ stdlib
import tomllib
with open("pyproject.toml", "rb") as f:
    config = tomllib.load(f)

# match/case (PEP 634) — Python 3.10+
match event:
    case ThreatDetectedEvent(confidence=c) if c > 0.9:
        send_critical_alert(event)
    case ThreatDetectedEvent():
        send_standard_alert(event)
    case _:
        pass
```

---

## Configuração de Projeto (pyproject.toml)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "meu-projeto"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111.0",
    "pydantic>=2.7.0",
    "sqlalchemy>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff", "mypy"]

[tool.ruff]
line-length = 99
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]  # UP = pyupgrade rules
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src --cov-report=term-missing --cov-fail-under=80"

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

---

## Checklist Python

- [ ] `python_requires` definido no `pyproject.toml`
- [ ] Todos os parâmetros e retornos com type hints
- [ ] Docstrings em todas as classes e métodos públicos (Google style)
- [ ] Logging com `logging.getLogger(__name__)` — nunca `print()`
- [ ] Exceções customizadas com hierarquia clara
- [ ] Imports organizados (stdlib → third-party → local)
- [ ] `ruff` e `mypy` configurados e passando
- [ ] `frozen=True` em dataclasses imutáveis
- [ ] Sem `except Exception: pass`
