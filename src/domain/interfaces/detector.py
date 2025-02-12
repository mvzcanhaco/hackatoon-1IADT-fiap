from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from ..entities.detection import DetectionResult

class DetectorInterface(ABC):
    """Interface base para detectores de objetos perigosos."""
    
    @abstractmethod
    def process_video(self, video_path: str, fps: int, threshold: float, resolution: int) -> Tuple[str, DetectionResult]:
        """Processa um vídeo e retorna as detecções encontradas."""
        pass
    
    @abstractmethod
    def clean_memory(self) -> None:
        """Limpa a memória utilizada pelo detector."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o dispositivo em uso."""
        pass
    
    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        pass 