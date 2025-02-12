from dataclasses import dataclass
from typing import Optional, Dict, Any
from ...domain.interfaces.detector import DetectorInterface
from ...domain.interfaces.notification import NotificationFactory
from ...domain.entities.detection import DetectionResult
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessVideoRequest:
    """DTO para requisição de processamento de vídeo."""
    video_path: str
    threshold: float = 0.5
    fps: Optional[int] = None
    resolution: Optional[int] = None
    notification_type: Optional[str] = None
    notification_target: Optional[str] = None

@dataclass
class ProcessVideoResponse:
    """DTO para resposta do processamento de vídeo."""
    status_message: str
    detection_result: DetectionResult
    memory_info: str
    device_info: str
    cache_stats: Optional[Dict[str, Any]] = None

class ProcessVideoUseCase:
    """Caso de uso para processamento de vídeo e notificação."""
    
    def __init__(
        self,
        detector: DetectorInterface,
        notification_factory: NotificationFactory,
        default_fps: int,
        default_resolution: int
    ):
        self.detector = detector
        self.notification_factory = notification_factory
        self.default_fps = default_fps
        self.default_resolution = default_resolution
    
    def execute(self, request: ProcessVideoRequest) -> ProcessVideoResponse:
        """Executa o processamento do vídeo e envia notificações se necessário."""
        try:
            # Usar valores padrão se não especificados
            fps = request.fps or self.default_fps
            resolution = request.resolution or self.default_resolution
            
            # Processar vídeo
            output_path, result = self.detector.process_video(
                request.video_path,
                fps=fps,
                threshold=request.threshold,
                resolution=resolution
            )
            
            # Enviar notificação se houver detecções e destino configurado
            if result.detections and request.notification_type and request.notification_target:
                notification_service = self.notification_factory.create_service(request.notification_type)
                if notification_service:
                    detection_data = {
                        'detections': [
                            {
                                'label': det.label,
                                'confidence': det.confidence,
                                'box': det.box,
                                'timestamp': det.timestamp
                            } for det in result.detections
                        ],
                        'technical': {
                            'threshold': request.threshold,
                            'fps': fps,
                            'resolution': resolution
                        }
                    }
                    notification_service.send_notification(detection_data, request.notification_target)
            
            # Formatar mensagem de status
            status_msg = self._format_status_message(result)
            
            # Obter informações do sistema de forma segura
            try:
                device_info = self.detector.get_device_info() if hasattr(self.detector, 'get_device_info') else {}
            except Exception as e:
                logger.error(f"Erro ao obter informações do dispositivo: {str(e)}")
                device_info = {}
                
            try:
                cache_stats = self.detector.get_cache_stats() if hasattr(self.detector, 'get_cache_stats') else {}
            except Exception as e:
                logger.error(f"Erro ao obter estatísticas do cache: {str(e)}")
                cache_stats = {}
            
            # Limpar memória
            try:
                self.detector.clean_memory()
            except Exception as e:
                logger.error(f"Erro ao limpar memória: {str(e)}")
            
            return ProcessVideoResponse(
                status_message=status_msg,
                detection_result=result,
                memory_info=self._format_memory_info(device_info),
                device_info=self._format_device_info(device_info),
                cache_stats=cache_stats
            )
            
        except Exception as e:
            logger.error(f"Erro ao executar caso de uso: {str(e)}")
            # Criar um resultado vazio em caso de erro
            empty_result = DetectionResult(
                video_path=request.video_path,
                detections=[],
                frames_analyzed=0,
                total_time=0.0,
                device_type="Unknown",
                frame_extraction_time=0.0,
                analysis_time=0.0
            )
            return ProcessVideoResponse(
                status_message="Erro ao processar o vídeo. Por favor, tente novamente.",
                detection_result=empty_result,
                memory_info="N/A",
                device_info="N/A",
                cache_stats={}
            )
    
    def _format_status_message(self, result: DetectionResult) -> str:
        """Formata a mensagem de status do processamento."""
        try:
            status = "⚠️ RISCO DETECTADO" if result.detections else "✅ SEGURO"
            
            message = f"""Processamento concluído! ({result.device_type})
            
Status: {status}
Detecções: {len(result.detections)}
Frames analisados: {result.frames_analyzed}
Tempo total: {result.total_time:.2f}s
Tempo de extração: {result.frame_extraction_time:.2f}s
Tempo de análise: {result.analysis_time:.2f}s"""

            # Adicionar detalhes das detecções se houver
            if result.detections:
                message += "\n\nDetecções encontradas:"
                for i, det in enumerate(result.detections[:3], 1):  # Mostrar até 3 detecções
                    message += f"\n{i}. {det.label} (Confiança: {det.confidence:.1%}, Frame: {det.frame})"
                if len(result.detections) > 3:
                    message += f"\n... e mais {len(result.detections) - 3} detecção(ões)"
            
            return message
            
        except Exception as e:
            logger.error(f"Erro ao formatar mensagem de status: {str(e)}")
            return "Erro ao processar o vídeo. Por favor, tente novamente."
    
    def _format_memory_info(self, device_info: Dict[str, Any]) -> str:
        if device_info.get('type') == 'GPU':
            return f"GPU: {device_info.get('memory_used', 0) / 1024**2:.1f}MB / {device_info.get('memory_total', 0) / 1024**2:.1f}MB"
        else:
            return f"RAM: {device_info.get('memory_used', 0) / 1024**2:.1f}MB / {device_info.get('memory_total', 0) / 1024**2:.1f}MB"
    
    def _format_device_info(self, device_info: Dict[str, Any]) -> str:
        if device_info.get('type') == 'GPU':
            return f"GPU: {device_info.get('name', 'Unknown')}"
        else:
            return f"CPU Threads: {device_info.get('threads', 'N/A')}" 