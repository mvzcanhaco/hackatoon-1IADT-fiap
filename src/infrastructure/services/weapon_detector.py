import torch
from typing import Tuple
from src.domain.interfaces.detector import DetectorInterface
from src.domain.entities.detection import Detection, DetectionResult
from src.domain.factories.detector_factory import WeaponDetector
from src.domain.detectors.gpu import WeaponDetectorGPU
from src.domain.detectors.cpu import WeaponDetectorCPU
import logging
import gc

logger = logging.getLogger(__name__)

class WeaponDetectorService(DetectorInterface):
    """Adaptador que conecta os detectores do domínio com a infraestrutura externa."""
    
    def __init__(self):
        try:
            # Usar o Factory Pattern do domínio para criar o detector apropriado
            self.detector = WeaponDetector.get_instance()  # Usar get_instance ao invés do construtor direto
            if not self.detector:
                raise RuntimeError("Falha ao criar o detector")
                
            self.device_type = "GPU" if torch.cuda.is_available() else "CPU"
            logger.info(f"Detector inicializado em modo {self.device_type}")
            
            # Manter referência à implementação específica para otimizações
            if hasattr(self.detector, '_instance') and self.detector._instance is not None:
                self._specific_detector = self.detector._instance
            else:
                self._specific_detector = self.detector
                
            # Verificar se o detector foi inicializado corretamente
            if not hasattr(self._specific_detector, 'process_video'):
                raise RuntimeError("Detector não possui método process_video")
                
            # Garantir que o detector está inicializado
            if hasattr(self._specific_detector, 'initialize'):
                self._specific_detector.initialize()
                
        except Exception as e:
            logger.error(f"Erro ao inicializar WeaponDetectorService: {str(e)}")
            raise RuntimeError(f"Falha na inicialização do detector: {str(e)}")
    
    def process_video(
        self,
        video_path: str,
        fps: int,
        threshold: float,
        resolution: int
    ) -> Tuple[str, DetectionResult]:
        """Processa o vídeo usando o detector apropriado."""
        try:
            if not self._specific_detector:
                raise RuntimeError("Detector não inicializado")
                
            # Garantir que o detector está inicializado
            if hasattr(self._specific_detector, 'initialize'):
                self._specific_detector.initialize()
                
            output_path, metrics = self._specific_detector.process_video(
                video_path,
                fps=fps,
                threshold=threshold,
                resolution=resolution
            )
            
            if not metrics:
                logger.warning("Nenhuma métrica retornada pelo detector")
                metrics = {}
            
            # Converter detecções para entidades do domínio
            detections = []
            for detection_group in metrics.get('detections', []):
                frame = detection_group.get('frame', 0)
                for det in detection_group.get('detections', []):
                    try:
                        detections.append(Detection(
                            frame=frame,
                            confidence=det.get('confidence', 0.0),
                            label=det.get('label', 'objeto perigoso'),  # Valor padrão mais informativo
                            box=det.get('box', [0, 0, 0, 0]),
                            timestamp=frame / fps if fps else 0
                        ))
                    except Exception as e:
                        logger.error(f"Erro ao processar detecção: {str(e)}")
            
            result = DetectionResult(
                video_path=output_path or video_path,
                detections=detections,
                frames_analyzed=metrics.get('frames_analyzed', 0),
                total_time=metrics.get('total_time', 0.0),
                device_type=self.device_type,
                frame_extraction_time=metrics.get('frame_extraction_time', 0.0),
                analysis_time=metrics.get('analysis_time', 0.0)
            )
            
            return output_path or video_path, result
            
        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            empty_result = DetectionResult(
                video_path=video_path,
                detections=[],
                frames_analyzed=0,
                total_time=0.0,
                device_type=self.device_type,
                frame_extraction_time=0.0,
                analysis_time=0.0
            )
            return video_path, empty_result
    
    def clean_memory(self) -> None:
        """Limpa a memória do detector."""
        try:
            if not self._specific_detector:
                logger.warning("Nenhum detector específico para limpar memória")
                return
                
            if hasattr(self._specific_detector, 'clear_cache'):
                self._specific_detector.clear_cache()
                
            if hasattr(self._specific_detector, 'clean_memory'):
                self._specific_detector.clean_memory()
                
            # Forçar coleta de lixo
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Erro ao limpar memória: {str(e)}")
    
    def get_device_info(self) -> dict:
        """Retorna informações detalhadas sobre o dispositivo em uso."""
        try:
            if not self._specific_detector:
                return {
                    "type": self.device_type,
                    "memory_total": 0,
                    "memory_used": 0
                }
                
            if isinstance(self._specific_detector, WeaponDetectorGPU):
                return {
                    "type": "GPU",
                    "name": torch.cuda.get_device_name(0),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory,
                    "memory_used": torch.cuda.memory_allocated(),
                    "memory_cached": torch.cuda.memory_reserved()
                }
            else:
                import psutil
                return {
                    "type": "CPU",
                    "threads": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "memory_used": psutil.virtual_memory().used
                }
        except Exception as e:
            logger.error(f"Erro ao obter informações do dispositivo: {str(e)}")
            return {
                "type": self.device_type,
                "memory_total": 0,
                "memory_used": 0
            }
    
    def get_cache_stats(self) -> dict:
        """Retorna estatísticas do cache se disponível."""
        try:
            if not self._specific_detector:
                return self._get_empty_cache_stats()
                
            if (hasattr(self._specific_detector, 'result_cache') and 
                self._specific_detector.result_cache is not None):
                return self._specific_detector.result_cache.get_stats()
                
            return self._get_empty_cache_stats()
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas do cache: {str(e)}")
            return self._get_empty_cache_stats()
            
    def _get_empty_cache_stats(self) -> dict:
        """Retorna estatísticas vazias do cache."""
        return {
            "cache_size": 0,
            "max_size": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": "0.00%",
            "memory_usage": 0
        } 