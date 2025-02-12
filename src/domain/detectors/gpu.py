import torch
import torch.nn.functional as F
import logging
import os
import gc
import numpy as np
import cv2
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from .base import BaseDetector

logger = logging.getLogger(__name__)

class WeaponDetectorGPU(BaseDetector):
    """Detector de armas otimizado para GPU."""
    
    def __init__(self):
        """Inicializa o detector."""
        super().__init__()
        self.default_resolution = 640
        self.device = None  # Será configurado em _initialize
        self._initialize()
    
    def _initialize(self):
        """Inicializa o modelo."""
        try:
            # Configurar device
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA não está disponível!")
            
            # Configurar device corretamente
            self.device = 0  # Usar índice inteiro para GPU
            
            # Carregar modelo e processador
            logger.info("Carregando modelo e processador...")
            model_name = "google/owlv2-base-patch16"
            
            self.owlv2_processor = Owlv2Processor.from_pretrained(model_name)
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": self.device}  # Mapear todo o modelo para GPU 0
            )
            
            # Otimizar modelo
            self.owlv2_model.eval()
            
            # Processar queries
            self.text_queries = self._get_detection_queries()
            self.processed_text = self.owlv2_processor(
                text=self.text_queries,
                return_tensors="pt",
                padding=True
            )
            self.processed_text = {
                key: val.to(self.device) 
                for key, val in self.processed_text.items()
            }
            
            logger.info("Inicialização GPU completa!")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Erro na inicialização GPU: {str(e)}")
            raise

    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> list:
        """Detecta objetos em uma imagem."""
        try:
            # Pré-processar imagem
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Processar imagem
            image_inputs = self.owlv2_processor(
                images=image,
                return_tensors="pt"
            )
            image_inputs = {
                key: val.to(self.device) 
                for key, val in image_inputs.items()
            }
            
            # Inferência
            with torch.no_grad():
                inputs = {**image_inputs, **self.processed_text}
                outputs = self.owlv2_model(**inputs)
                
                target_sizes = torch.tensor([image.size[::-1]], device=self.device)
                results = self.owlv2_processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=threshold
                )[0]
            
            # Processar detecções
            detections = []
            if len(results["scores"]) > 0:
                scores = results["scores"]
                boxes = results["boxes"]
                labels = results["labels"]
                
                for score, box, label in zip(scores, boxes, labels):
                    if score.item() >= threshold:
                        detections.append({
                            "confidence": score.item(),
                            "box": [int(x) for x in box.tolist()],
                            "label": self.text_queries[label]
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Erro em detect_objects: {str(e)}")
            return []

    def _get_best_device(self):
        """Retorna o melhor dispositivo disponível."""
        return 0  # Usar índice inteiro para GPU

    def _clear_gpu_memory(self):
        """Limpa memória GPU."""
        torch.cuda.empty_cache()
        gc.collect()

    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3, resolution: int = 640) -> tuple:
        """Processa um vídeo."""
        metrics = {
            "total_time": 0,
            "frames_analyzed": 0,
            "detections": []
        }
        
        try:
            frames = self.extract_frames(video_path, fps or 2, resolution)
            metrics["frames_analyzed"] = len(frames)
            
            for i, frame in enumerate(frames):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                detections = self.detect_objects(frame_pil, threshold)
                if detections:
                    metrics["detections"].append({
                        "frame": i,
                        "detections": detections
                    })
                    return video_path, metrics
            
            return video_path, metrics
            
        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            return video_path, metrics