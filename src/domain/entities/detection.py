from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Detection:
    """Representa uma detecção de objeto perigoso."""
    frame: int
    confidence: float
    label: str
    box: List[int]  # [x1, y1, x2, y2]
    timestamp: float = 0.0

@dataclass
class DetectionResult:
    """Resultado completo do processamento de vídeo."""
    video_path: str
    detections: List[Detection]
    frames_analyzed: int
    total_time: float
    device_type: str
    frame_extraction_time: float
    analysis_time: float