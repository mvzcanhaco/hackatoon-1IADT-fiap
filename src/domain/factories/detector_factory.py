import os
import time
import gc
import cv2
import json
import torch
import psutil
import shutil
import pickle
import hashlib
import tempfile
import logging
import subprocess
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from dotenv import load_dotenv
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import Optional
from src.domain.detectors.base import BaseDetector, BaseCache
from src.domain.detectors.gpu import WeaponDetectorGPU
from src.domain.detectors.cpu import WeaponDetectorCPU

# Carregar variáveis de ambiente
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_gpu_init():
    """Força a inicialização da GPU."""
    try:
        # Verificar se CUDA está disponível
        if not torch.cuda.is_available():
            return False
            
        # Tentar alocar um tensor na GPU
        try:
            dummy = torch.cuda.FloatTensor(1)
            del dummy
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            return False
            
    except Exception as e:
        logger.warning(f"Erro ao forçar inicialização da GPU: {str(e)}")
        return False

def is_gpu_available():
    """Verifica se a GPU está disponível de forma mais robusta."""
    try:
        # Verificar CUDA primeiro
        if not torch.cuda.is_available():
            logger.warning("CUDA não está disponível")
            return False
        
        # Tentar forçar inicialização
        if not force_gpu_init():
            logger.warning("Não foi possível inicializar a GPU")
            return False
            
        # Tentar obter informações da GPU
        try:
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.warning("Nenhuma GPU encontrada")
                return False
                
            # Verificar se podemos realmente usar a GPU
            device = torch.device(0)  # Usar índice do dispositivo
            dummy_tensor = torch.zeros(1, device=device)
            del dummy_tensor
            torch.cuda.empty_cache()
            
            logger.info(f"GPU disponível: {torch.cuda.get_device_name(0)}")
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao verificar GPU: {str(e)}")
            return False
            
    except Exception as e:
        logger.warning(f"Erro ao verificar disponibilidade da GPU: {str(e)}")
        return False

class BaseCache:
    """Cache base para armazenar resultados de detecção."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.last_access = {}

    def get(self, image: np.ndarray) -> list:
        try:
            key = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
            if key in self.cache:
                self.hits += 1
                self.last_access[key] = time.time()
                return self.cache[key]
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Erro ao recuperar do cache: {str(e)}")
            return None

    def put(self, image: np.ndarray, results: list):
        try:
            key = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
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

class BaseWeaponDetector:
    """Classe base abstrata para detecção de armas."""
    def __init__(self):
        """Inicialização básica comum a todos os detectores."""
        self._initialized = False
        self.device = self._get_best_device()
        self._initialize()

    def _check_initialized(self):
        """Verifica se o detector está inicializado."""
        if not self._initialized:
            raise RuntimeError("Detector não está inicializado")

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

    def _get_best_device(self):
        """Deve ser implementado nas classes filhas."""
        raise NotImplementedError

    def _initialize(self):
        """Deve ser implementado nas classes filhas."""
        raise NotImplementedError

    def detect_objects(self, image, threshold=0.3):
        """Deve ser implementado nas classes filhas."""
        raise NotImplementedError

    def process_video(self, video_path, fps=None, threshold=0.3, resolution=640):
        """Deve ser implementado nas classes filhas."""
        raise NotImplementedError

    def _apply_nms(self, detections: list, iou_threshold: float = 0.5) -> list:
        """Deve ser implementado nas classes filhas."""
        raise NotImplementedError

    def extract_frames(self, video_path: str, fps: int = 2, resolution: int = 640) -> list:
        """Extrai frames de um vídeo utilizando ffmpeg."""
        frames = []
        temp_dir = Path(tempfile.mkdtemp())
        try:
            threads = min(os.cpu_count(), 8)
            cmd = [
                'ffmpeg', '-i', video_path,
                '-threads', str(threads),
                '-vf', (f'fps={fps},'
                        f'scale={resolution}:{resolution}:force_original_aspect_ratio=decrease:flags=lanczos,'
                        f'pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2'),
                '-frame_pts', '1',
                f'{temp_dir}/%d.jpg'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            frame_files = sorted(temp_dir.glob('*.jpg'), key=lambda x: int(x.stem))
            chunk_size = 100
            with ThreadPoolExecutor(max_workers=threads) as executor:
                for i in range(0, len(frame_files), chunk_size):
                    chunk = frame_files[i:i + chunk_size]
                    chunk_frames = list(tqdm(
                        executor.map(lambda f: cv2.imread(str(f)), chunk),
                        desc=f"Carregando frames {i+1}-{min(i+chunk_size, len(frame_files))}",
                        total=len(chunk)
                    ))
                    frames.extend(chunk_frames)
                    if i % (chunk_size * 5) == 0:
                        gc.collect()
        finally:
            shutil.rmtree(temp_dir)
        return frames

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Deve ser implementado nas classes filhas."""
        raise NotImplementedError

    def _update_frame_metrics(self, detections: list, frame_idx: int, metrics: dict):
        """Atualiza as métricas para um conjunto de detecções em um frame."""
        try:
            for detection in detections:
                self._update_detection_metrics(detection, metrics)
                if isinstance(detection, dict):
                    metrics.setdefault("detections", []).append({
                        "frame": frame_idx,
                        "box": detection.get("box", []),
                        "confidence": detection.get("confidence", 0),
                        "label": detection.get("label", "unknown")
                    })
        except Exception as e:
            logger.error(f"Erro ao atualizar métricas do frame: {str(e)}")

    def _update_detection_metrics(self, detection: dict, metrics: dict):
        """Atualiza as métricas de detecção."""
        try:
            if not isinstance(detection, dict):
                logger.warning(f"Detection não é um dicionário: {detection}")
                return
            confidence = detection.get("confidence", 0)
            if not confidence:
                return
            if "detection_stats" not in metrics:
                metrics["detection_stats"] = {
                    "total_detections": 0,
                    "avg_confidence": 0,
                    "confidence_distribution": {
                        "low": 0,
                        "medium": 0,
                        "high": 0
                    }
                }
            stats = metrics["detection_stats"]
            stats["total_detections"] += 1
            if confidence < 0.5:
                stats["confidence_distribution"]["low"] += 1
            elif confidence < 0.7:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["high"] += 1
            n = stats["total_detections"]
            old_avg = stats["avg_confidence"]
            stats["avg_confidence"] = (old_avg * (n - 1) + confidence) / n
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def clear_cache(self):
        """Deve ser implementado nas classes filhas."""
        raise NotImplementedError

class ResultCache(BaseCache):
    """
    Cache otimizado para armazenar resultados de detecção.
    """
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size)

class WeaponDetector:
    """Implementação do Factory Pattern para criar a instância apropriada do detector."""
    _instance = None

    def __new__(cls):
        try:
            if cls._instance is None:
                if torch.cuda.is_available():
                    cls._instance = WeaponDetectorGPU()
                    logger.info("Detector GPU criado")
                else:
                    cls._instance = WeaponDetectorCPU()
                    logger.info("Detector CPU criado")
                
                # Garantir que o detector foi inicializado corretamente
                if not cls._instance:
                    raise RuntimeError("Falha ao criar instância do detector")
                    
                # Inicializar o detector
                if hasattr(cls._instance, 'initialize'):
                    cls._instance.initialize()
                    
                # Verificar se os métodos necessários existem
                required_methods = ['process_video', 'clean_memory', 'detect_objects']
                for method in required_methods:
                    if not hasattr(cls._instance, method):
                        raise RuntimeError(f"Detector não possui método obrigatório: {method}")
                
            return cls._instance
            
        except Exception as e:
            logger.error(f"Erro ao criar detector: {str(e)}")
            raise

    @classmethod
    def get_instance(cls):
        """Retorna a instância existente ou cria uma nova."""
        if cls._instance is None:
            return cls()
        return cls._instance

    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> list:
        """Detecta objetos em uma imagem."""
        if not self._instance:
            raise RuntimeError("Detector não inicializado")
        return self._instance.detect_objects(image, threshold)

    def extract_frames(self, video_path: str, fps: int = 2, resolution: int = 640) -> list:
        """Extrai frames de um vídeo."""
        if not self._instance:
            raise RuntimeError("Detector não inicializado")
        return self._instance.extract_frames(video_path, fps, resolution)

    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3, resolution: int = 640) -> tuple:
        """Processa o vídeo e retorna os detalhes técnicos e as detecções."""
        if not self._instance:
            raise RuntimeError("Detector não inicializado")
        return self._instance.process_video(video_path, fps, threshold, resolution)

    def clean_memory(self):
        """Limpa todo o cache do sistema."""
        if not self._instance:
            return
        if hasattr(self._instance, 'clear_cache'):
            self._instance.clear_cache()
        if hasattr(self._instance, 'clean_memory'):
            self._instance.clean_memory()
        # Forçar limpeza de memória
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class DetectorFactory:
    """Factory para criar a instância apropriada do detector."""
    
    @staticmethod
    def create_detector() -> BaseDetector:
        """Cria e retorna a instância apropriada do detector."""
        try:
            # Forçar verificação robusta de GPU
            if is_gpu_available():
                logger.info("GPU disponível e inicializada com sucesso")
                return WeaponDetectorGPU()
            else:
                logger.warning("GPU não disponível ou com problemas. ATENÇÃO: O sistema funcionará em modo CPU, " +
                             "que é mais lento mas igualmente funcional. Performance será reduzida.")
                return WeaponDetectorCPU()
        except Exception as e:
            logger.error(f"Erro ao criar detector: {str(e)}")
            logger.warning("Fallback para CPU devido a erro. O sistema continuará funcionando, " +
                         "mas com performance reduzida.")
            return WeaponDetectorCPU()