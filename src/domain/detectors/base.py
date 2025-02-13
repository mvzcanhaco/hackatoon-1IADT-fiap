from abc import ABC, abstractmethod
import gc
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
import os
import cv2
from PIL import Image
import time
import sys

logger = logging.getLogger(__name__)

class BaseCache:
    """Cache base para armazenar resultados de detecção."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.last_access = {}

    def get(self, key: str) -> Optional[Dict]:
        try:
            if key in self.cache:
                self.hits += 1
                self.last_access[key] = time.time()
                return self.cache[key]
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Erro ao recuperar do cache: {str(e)}")
            return None

    def put(self, key: str, results: Dict):
        try:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.last_access.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.last_access[oldest_key]
            self.cache[key] = results
            self.last_access[key] = time.time()
        except Exception as e:
            logger.error(f"Erro ao armazenar no cache: {str(e)}")

    def clear(self):
        """Limpa o cache e libera memória."""
        self.cache.clear()
        self.last_access.clear()
        gc.collect()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "memory_usage": sum(sys.getsizeof(v) for v in self.cache.values())
        }

class BaseDetector(ABC):
    """Classe base abstrata para detectores de objetos perigosos."""
    def __init__(self):
        self._initialized = False
        self.device = None
        self.owlv2_model = None
        self.owlv2_processor = None
        self.text_queries = None
        self.processed_text = None
        self.threshold = 0.3
        self.result_cache = None
        
    @abstractmethod
    def _initialize(self):
        """Inicializa o modelo e o processador."""
        pass
        
    @abstractmethod
    def _get_best_device(self):
        """Retorna o melhor dispositivo disponível."""
        pass
        
    def initialize(self):
        """Inicializa o detector se ainda não estiver inicializado."""
        if not self._initialized:
            self._initialize()
            
    def extract_frames(self, video_path: str, fps: int = None, resolution: int = 640) -> List:
        """Extrai frames do vídeo com taxa e resolução especificadas."""
        try:
            if not os.path.exists(video_path):
                logger.error(f"Arquivo de vídeo não encontrado: {video_path}")
                return []
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Erro ao abrir o vídeo")
                return []
                
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = fps if fps else min(2, original_fps)
            frame_interval = int(original_fps / target_fps)
            
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    if resolution:
                        height, width = frame.shape[:2]
                        scale = resolution / max(height, width)
                        if scale < 1:
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                    frames.append(frame)
                    
                frame_count += 1
                
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Erro ao extrair frames: {str(e)}")
            return []
            
    @abstractmethod
    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> List[Dict]:
        """Detecta objetos em uma imagem."""
        pass
        
    @abstractmethod
    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3, resolution: int = 640) -> Tuple[str, Dict]:
        """Processa um vídeo para detecção de objetos."""
        pass

    def clean_memory(self):
        """Limpa memória não utilizada."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cache GPU limpo")
            gc.collect()
            logger.debug("Garbage collector executado")
        except Exception as e:
            logger.error(f"Erro ao limpar memória: {str(e)}")

    def _get_detection_queries(self) -> List[str]:
        """Retorna as queries otimizadas para detecção de objetos perigosos."""
        firearms = ["handgun", "rifle", "shotgun", "machine gun", "firearm"]
        edged_weapons = ["knife", "dagger", "machete", "box cutter", "sword"]
        ranged_weapons = ["crossbow", "bow","arrow"]
        sharp_objects = ["blade", "razor", "glass shard", "screwdriver", "metallic pointed object"]
        
        firearm_contexts = ["close-up", "clear view", "detailed"]
        edged_contexts = ["close-up", "clear view", "detailed", "metallic", "sharp"]
        ranged_contexts = ["close-up", "clear view", "detailed"]
        sharp_contexts = ["close-up", "clear view", "detailed", "sharp"]
        
        queries = []
        
        for weapon in firearms:
            queries.append(f"a photo of a {weapon}")
            for context in firearm_contexts:
                queries.append(f"a photo of a {context} {weapon}")
        
        for weapon in edged_weapons:
            queries.append(f"a photo of a {weapon}")
            for context in edged_contexts:
                queries.append(f"a photo of a {context} {weapon}")
        
        for weapon in ranged_weapons:
            queries.append(f"a photo of a {weapon}")
            for context in ranged_contexts:
                queries.append(f"a photo of a {context} {weapon}")
        
        for weapon in sharp_objects:
            queries.append(f"a photo of a {weapon}")
            for context in sharp_contexts:
                queries.append(f"a photo of a {context} {weapon}")
        
        queries = sorted(list(set(queries)))
        logger.info(f"Total de queries otimizadas geradas: {len(queries)}")
        return queries

    @abstractmethod
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Aplica Non-Maximum Suppression nas detecções."""
        pass

    @abstractmethod
    def _preprocess_image(self, image: Any) -> Any:
        """Pré-processa a imagem para o formato adequado."""
        pass 