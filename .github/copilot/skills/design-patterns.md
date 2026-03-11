# Skill: Design Patterns

Catálogo de padrões aplicados no contexto de arquitetura hexagonal e DDD.

---

## Padrões de Criação

### Factory Method

**Quando usar**: Criação de objetos cuja implementação exata é determinada em runtime.

**No contexto da arquitetura hexagonal**: Factories ficam no domínio ou infraestrutura, nunca expõem detalhes de implementação.

```python
# Exemplo real deste projeto: DetectorFactory
from abc import ABC, abstractmethod
from ..interfaces.detector import DetectorInterface

class DetectorFactory(ABC):
    """Port factory — define o contrato de criação."""

    @abstractmethod
    def create(self) -> DetectorInterface:
        ...


class GpuDetectorFactory(DetectorFactory):
    """Cria detector otimizado para GPU."""

    def create(self) -> DetectorInterface:
        from ...infrastructure.services.gpu_detector import GpuDetector
        return GpuDetector(device="cuda")


class CpuDetectorFactory(DetectorFactory):
    """Cria detector para CPU — fallback quando GPU não disponível."""

    def create(self) -> DetectorInterface:
        from ...infrastructure.services.cpu_detector import CpuDetector
        return CpuDetector(threads=os.cpu_count())


# Bootstrap — decide qual factory usar:
def create_detector_factory() -> DetectorFactory:
    if torch.cuda.is_available():
        return GpuDetectorFactory()
    return CpuDetectorFactory()
```

---

### Abstract Factory

**Quando usar**: Criar famílias de objetos relacionados sem especificar classes concretas.

```python
class NotificationFactory(ABC):
    """Abstract Factory para serviços de notificação."""

    @abstractmethod
    def create_service(self, notification_type: str) -> Optional[NotificationInterface]:
        ...

class ProductionNotificationFactory(NotificationFactory):
    def create_service(self, notification_type: str) -> Optional[NotificationInterface]:
        services = {
            "webhook": WebhookNotificationService(),
            "email": SmtpEmailService(),
            "sms": TwilioSmsService(),
        }
        return services.get(notification_type)

class TestNotificationFactory(NotificationFactory):
    """Factory para testes — todos os serviços são mocks."""
    def create_service(self, notification_type: str) -> Optional[NotificationInterface]:
        return InMemoryNotificationService()
```

---

### Builder

**Quando usar**: Construção passo-a-passo de objetos complexos. Muito útil em testes.

```python
class DetectionResultBuilder:
    """Builder para DetectionResult — útil em testes."""

    def __init__(self):
        self._video_path = "test.mp4"
        self._detections: List[Detection] = []
        self._frames_analyzed = 0
        self._total_time = 0.0
        self._device_type = "CPU"

    def with_video_path(self, path: str) -> "DetectionResultBuilder":
        self._video_path = path
        return self

    def with_detection(self, label: str, confidence: float) -> "DetectionResultBuilder":
        self._detections.append(Detection(label=label, confidence=confidence))
        return self

    def with_n_detections(self, n: int) -> "DetectionResultBuilder":
        for i in range(n):
            self._detections.append(Detection(label=f"weapon_{i}", confidence=0.9))
        return self

    def on_gpu(self) -> "DetectionResultBuilder":
        self._device_type = "GPU"
        return self

    def build(self) -> DetectionResult:
        return DetectionResult(
            video_path=self._video_path,
            detections=self._detections,
            frames_analyzed=self._frames_analyzed,
            total_time=self._total_time,
            device_type=self._device_type,
        )
```

---

## Padrões Estruturais

### Adapter

**Quando usar**: Converter a interface de uma classe em outra que o cliente espera.

**No contexto hexagonal**: É o coração da camada de infraestrutura.

```python
# Port (interface esperada pelo use case)
class PaymentGatewayPort(ABC):
    @abstractmethod
    def charge(self, amount: Money, card_token: str) -> PaymentResult:
        ...

# Adapter (converte chamada do nosso domínio para API externa)
class StripePaymentAdapter(PaymentGatewayPort):
    """Adapta a API do Stripe para o Port de pagamento do domínio."""

    def __init__(self, stripe_client: stripe.Stripe) -> None:
        self._stripe = stripe_client

    def charge(self, amount: Money, card_token: str) -> PaymentResult:
        try:
            charge = self._stripe.PaymentIntent.create(
                amount=int(amount.in_cents()),
                currency=amount.currency.lower(),
                payment_method=card_token,
            )
            return PaymentResult(
                transaction_id=TransactionId(charge.id),
                status=PaymentStatus.APPROVED,
            )
        except stripe.error.CardError as e:
            return PaymentResult(
                transaction_id=None,
                status=PaymentStatus.DECLINED,
                reason=str(e),
            )
```

---

### Facade

**Quando usar**: Simplificar uma interface complexa, agrupando operações relacionadas.

```python
class DetectionFacade:
    """
    Facade que simplifica o processo completo de detecção.
    Esconde a complexidade de orquestrar detector + notificação + métricas.
    """

    def __init__(
        self,
        detector_factory: DetectorFactory,
        notification_factory: NotificationFactory,
        metrics_recorder: MetricsPort,
    ) -> None:
        self._detector_factory = detector_factory
        self._notification_factory = notification_factory
        self._metrics = metrics_recorder

    def detect_and_notify(self, video_path: str, config: DetectionConfig) -> DetectionResult:
        """Um único método que orquestra tudo."""
        detector = self._detector_factory.create()
        result = detector.process_video(video_path, config.fps, config.threshold)
        if result.has_threats() and config.notification_target:
            notifier = self._notification_factory.create_service(config.notification_type)
            notifier.send(result, config.notification_target)
        self._metrics.record(result)
        return result
```

---

### Decorator

**Quando usar**: Adicionar comportamento a objetos sem modificar sua classe.

```python
class LoggingDetectorDecorator(DetectorInterface):
    """Adiciona logging ao detector sem modificar a implementação original."""

    def __init__(self, wrapped: DetectorInterface) -> None:
        self._wrapped = wrapped
        self._logger = logging.getLogger(__name__)

    def process_video(self, video_path: str, fps: int, threshold: float, resolution: int):
        self._logger.info(f"Starting detection: {video_path}")
        start = time.monotonic()
        result = self._wrapped.process_video(video_path, fps, threshold, resolution)
        elapsed = time.monotonic() - start
        self._logger.info(f"Detection completed in {elapsed:.2f}s. Threats: {len(result.detections)}")
        return result

    def clean_memory(self) -> None:
        self._wrapped.clean_memory()

    def get_device_info(self):
        return self._wrapped.get_device_info()

    def get_cache_stats(self):
        return self._wrapped.get_cache_stats()


class CachingDetectorDecorator(DetectorInterface):
    """Adiciona cache de resultados ao detector."""
    ...
```

---

## Padrões Comportamentais

### Strategy

**Quando usar**: Família de algoritmos intercambiáveis com a mesma interface.

```python
# Interface da estratégia (port no domínio)
class VideoProcessingStrategy(ABC):
    @abstractmethod
    def extract_frames(self, video_path: str, fps: int) -> List[VideoFrame]:
        ...

# Estratégias concretas (adapters na infra)
class OpenCvFrameExtractor(VideoProcessingStrategy):
    def extract_frames(self, video_path: str, fps: int) -> List[VideoFrame]:
        ...

class FFmpegFrameExtractor(VideoProcessingStrategy):
    def extract_frames(self, video_path: str, fps: int) -> List[VideoFrame]:
        ...

# Contexto usa a estratégia via interface
class VideoProcessor:
    def __init__(self, strategy: VideoProcessingStrategy) -> None:
        self._strategy = strategy

    def process(self, video_path: str, fps: int) -> ProcessingResult:
        frames = self._strategy.extract_frames(video_path, fps)
        ...
```

---

### Observer / Event Dispatcher

**Quando usar**: Notificar múltiplos interessados quando algo acontece no domínio.

```python
from typing import Callable, Dict, List, Type

DomainEventHandler = Callable[[DomainEvent], None]

class DomainEventDispatcher:
    """Dispatches domain events to registered handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[Type[DomainEvent], List[DomainEventHandler]] = {}

    def subscribe(self, event_type: Type[DomainEvent], handler: DomainEventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def dispatch(self, event: DomainEvent) -> None:
        for handler in self._handlers.get(type(event), []):
            handler(event)


# Uso no use case:
class ProcessVideoUseCase:
    def __init__(self, ..., event_dispatcher: DomainEventDispatcher):
        self._dispatcher = event_dispatcher

    def execute(self, request: ProcessVideoRequest) -> ProcessVideoResponse:
        ...
        if result.has_threats():
            self._dispatcher.dispatch(ThreatDetectedEvent(
                video_path=request.video_path,
                detections=result.detections,
            ))
```

---

### Repository Pattern

**Quando usar**: Sempre, para abstrair persistência. Ver skill de hexagonal architecture.

```python
# Specification Pattern — combinado com Repository para queries expressivas
class Specification(ABC):
    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        ...

    def and_(self, other: "Specification") -> "AndSpecification":
        return AndSpecification(self, other)

    def or_(self, other: "Specification") -> "OrSpecification":
        return OrSpecification(self, other)

class HighConfidenceDetectionSpec(Specification[Detection]):
    def __init__(self, min_confidence: float = 0.8):
        self._min = min_confidence

    def is_satisfied_by(self, detection: Detection) -> bool:
        return detection.confidence >= self._min

# Uso:
high_confidence = HighConfidenceDetectionSpec(0.8)
weapon_spec = WeaponTypeSpec("knife")
dangerous = high_confidence.and_(weapon_spec)
threats = [d for d in result.detections if dangerous.is_satisfied_by(d)]
```

---

## Guia Rápido — Qual Pattern Usar?

| Situação | Pattern |
|----------|---------|
| Múltiplas implementações de um algoritmo | Strategy |
| Criação complexa com configurações variadas | Builder |
| Decisão de qual objeto criar em runtime | Factory Method |
| Famílias de objetos relacionados | Abstract Factory |
| Converter interface de terceiro para domínio | Adapter |
| Adicionar comportamento sem herança | Decorator |
| Simplificar subsistema complexo | Facade |
| Notificar múltiplos interessados | Observer |
| Encapsular regra de seleção/filtragem | Specification |
| Desfazer operações | Command + Memento |
| Cache transparente | Proxy |

---

## Quando Usar Este Skill

```
@workspace #file:.github/copilot/skills/design-patterns.md
Qual pattern devo usar para [descreva o problema]?
```
