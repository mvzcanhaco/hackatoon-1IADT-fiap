import torch
import torch.nn.functional as F
import logging
import os
import gc
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple
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
            self.device = torch.device("cuda:0")  # Usar device CUDA
            
            # Carregar modelo e processador
            logger.info("Carregando modelo e processador...")
            model_name = "google/owlv2-base-patch16"
            
            self.owlv2_processor = Owlv2Processor.from_pretrained(model_name)
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": 0}  # Mapear todo o modelo para GPU 0
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

    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> List[Dict]:
        """Detecta objetos em uma imagem."""
        try:
            # Pré-processar imagem
            image = self._preprocess_image(image)
            
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
            
            # Aplicar NMS nas detecções
            detections = self._apply_nms(detections)
            return detections
            
        except Exception as e:
            logger.error(f"Erro em detect_objects: {str(e)}")
            return []

    def _get_best_device(self) -> torch.device:
        """Retorna o melhor dispositivo disponível."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _clear_gpu_memory(self):
        """Limpa memória GPU."""
        torch.cuda.empty_cache()
        gc.collect()

    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3, resolution: int = 640) -> Tuple[str, Dict]:
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

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Pré-processa a imagem para o formato esperado pelo modelo."""
        try:
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Redimensionar mantendo proporção
            target_size = (self.default_resolution, self.default_resolution)
            if image.size != target_size:
                ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Adicionar padding se necessário
                if new_size != target_size:
                    new_image = Image.new('RGB', target_size, (0, 0, 0))
                    paste_x = (target_size[0] - new_size[0]) // 2
                    paste_y = (target_size[1] - new_size[1]) // 2
                    new_image.paste(image, (paste_x, paste_y))
                    image = new_image

            return image
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {str(e)}")
            return image

    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Aplica Non-Maximum Suppression nas detecções."""
        try:
            if not detections or len(detections) <= 1:
                return detections

            # Extrair scores e boxes
            scores = torch.tensor([d["confidence"] for d in detections], device=self.device)
            boxes = torch.tensor([[d["box"][0], d["box"][1], d["box"][2], d["box"][3]] 
                                for d in detections], device=self.device)

            # Ordenar por score
            _, order = scores.sort(descending=True)
            keep = []

            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break

                i = order[0]
                keep.append(i.item())

                # Calcular IoU com os boxes restantes
                box1 = boxes[i]
                box2 = boxes[order[1:]]
                
                # Calcular interseção
                left = torch.max(box1[0], box2[:, 0])
                top = torch.max(box1[1], box2[:, 1])
                right = torch.min(box1[2], box2[:, 2])
                bottom = torch.min(box1[3], box2[:, 3])
                
                width = torch.clamp(right - left, min=0)
                height = torch.clamp(bottom - top, min=0)
                inter = width * height

                # Calcular união
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
                union = area1 + area2 - inter

                # Calcular IoU
                iou = inter / union
                mask = iou <= iou_threshold
                order = order[1:][mask]

            # Retornar detecções filtradas
            return [detections[i] for i in keep]

        except Exception as e:
            logger.error(f"Erro ao aplicar NMS: {str(e)}")
            return detections