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
import time

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
            logger.info(f"Queries carregadas: {self.text_queries}")  # Log das queries
            
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
                    score_val = score.item()
                    if score_val >= threshold:
                        # Garantir que o índice está dentro dos limites
                        label_idx = min(label.item(), len(self.text_queries) - 1)
                        label_text = self.text_queries[label_idx]
                        detections.append({
                            "confidence": round(score_val * 100, 2),  # Converter para porcentagem
                            "box": [int(x) for x in box.tolist()],
                            "label": label_text
                        })
                        logger.debug(f"Detecção: {label_text} ({score_val * 100:.2f}%)")
            
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
        metrics = {
            "total_time": 0,
            "frame_extraction_time": 0,
            "analysis_time": 0,
            "frames_analyzed": 0,
            "video_duration": 0,
            "device_type": "GPU",
            "detections": []
        }
        
        try:
            start_time = time.time()
            
            # Extrair frames
            t0 = time.time()
            frames = self.extract_frames(video_path, fps or 2, resolution)
            metrics["frame_extraction_time"] = time.time() - t0
            metrics["frames_analyzed"] = len(frames)
            
            if not frames:
                logger.warning("Nenhum frame extraído do vídeo")
                return video_path, metrics
            
            # Calcular duração do vídeo
            metrics["video_duration"] = len(frames) / (fps or 2)
            
            # Processar frames em batch
            t0 = time.time()
            batch_size = 2  # Reduzido ainda mais para garantir compatibilidade
            detections_by_frame = []
            
            for i in range(0, len(frames), batch_size):
                try:
                    batch_frames = frames[i:i + batch_size]
                    batch_pil_frames = []
                    
                    # Preparar batch
                    for frame in batch_frames:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        frame_pil = self._preprocess_image(frame_pil)
                        batch_pil_frames.append(frame_pil)
                    
                    # Processar batch
                    batch_inputs = self.owlv2_processor(
                        images=batch_pil_frames,
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Validar shapes antes da inferência
                    if not self._validate_batch_shapes(batch_inputs):
                        logger.warning(f"Shape inválido detectado no batch {i}, processando frames individualmente...")
                        # Processar frames individualmente
                        for frame_idx, frame_pil in enumerate(batch_pil_frames):
                            try:
                                single_input = self.owlv2_processor(
                                    images=frame_pil,
                                    return_tensors="pt"
                                )
                                single_input = {
                                    key: val.to(self.device) 
                                    for key, val in single_input.items()
                                }
                                
                                with torch.no_grad():
                                    inputs = {**single_input, **self.processed_text}
                                    outputs = self.owlv2_model(**inputs)
                                    
                                    target_sizes = torch.tensor([frame_pil.size[::-1]], device=self.device)
                                    results = self.owlv2_processor.post_process_grounded_object_detection(
                                        outputs=outputs,
                                        target_sizes=target_sizes,
                                        threshold=threshold
                                    )
                                    
                                    if len(results[0]["scores"]) > 0:
                                        scores = results[0]["scores"]
                                        boxes = results[0]["boxes"]
                                        labels = results[0]["labels"]
                                        
                                        frame_detections = []
                                        for score, box, label in zip(scores, boxes, labels):
                                            score_val = score.item()
                                            if score_val >= threshold:
                                                label_idx = min(label.item(), len(self.text_queries) - 1)
                                                label_text = self.text_queries[label_idx]
                                                frame_detections.append({
                                                    "confidence": round(score_val * 100, 2),
                                                    "box": [int(x) for x in box.tolist()],
                                                    "label": label_text,
                                                    "frame": i + frame_idx,
                                                    "timestamp": (i + frame_idx) / (fps or 2)
                                                })
                                        
                                        if frame_detections:
                                            frame_detections = self._apply_nms(frame_detections)
                                            detections_by_frame.extend(frame_detections)
                                            
                            except Exception as e:
                                logger.error(f"Erro ao processar frame individual {i + frame_idx}: {str(e)}")
                                continue
                                
                            finally:
                                if 'single_input' in locals():
                                    del single_input
                                if 'outputs' in locals():
                                    del outputs
                                torch.cuda.empty_cache()
                        continue
                    
                    # Processar batch normalmente
                    batch_inputs = {
                        key: val.to(self.device) 
                        for key, val in batch_inputs.items()
                    }
                    
                    with torch.no_grad():
                        inputs = {**batch_inputs, **self.processed_text}
                        outputs = self.owlv2_model(**inputs)
                        
                        target_sizes = torch.tensor(
                            [frame.size[::-1] for frame in batch_pil_frames],
                            device=self.device
                        )
                        results = self.owlv2_processor.post_process_grounded_object_detection(
                            outputs=outputs,
                            target_sizes=target_sizes,
                            threshold=threshold
                        )
                    
                    # Processar resultados do batch
                    for frame_idx, frame_results in enumerate(results):
                        if len(frame_results["scores"]) > 0:
                            scores = frame_results["scores"]
                            boxes = frame_results["boxes"]
                            labels = frame_results["labels"]
                            
                            frame_detections = []
                            for score, box, label in zip(scores, boxes, labels):
                                score_val = score.item()
                                if score_val >= threshold:
                                    label_idx = min(label.item(), len(self.text_queries) - 1)
                                    label_text = self.text_queries[label_idx]
                                    frame_detections.append({
                                        "confidence": round(score_val * 100, 2),
                                        "box": [int(x) for x in box.tolist()],
                                        "label": label_text,
                                        "frame": i + frame_idx,
                                        "timestamp": (i + frame_idx) / (fps or 2)
                                    })
                            
                            if frame_detections:
                                frame_detections = self._apply_nms(frame_detections)
                                detections_by_frame.extend(frame_detections)
                
                except RuntimeError as e:
                    logger.error(f"Erro no processamento do batch {i}: {str(e)}")
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                    
                finally:
                    # Liberar memória do batch
                    del batch_inputs
                    if 'outputs' in locals():
                        del outputs
                    torch.cuda.empty_cache()
            
            # Atualizar métricas finais
            metrics["analysis_time"] = time.time() - t0
            metrics["total_time"] = time.time() - start_time
            metrics["detections"] = detections_by_frame
            
            return video_path, metrics
            
        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            return video_path, metrics

    def _validate_batch_shapes(self, batch_inputs: Dict) -> bool:
        """Valida os shapes dos tensores do batch."""
        try:
            pixel_values = batch_inputs.get("pixel_values")
            if pixel_values is None:
                return False
                
            batch_size = pixel_values.shape[0]
            if batch_size == 0:
                return False
                
            # Validar dimensões esperadas
            expected_dims = 4  # [batch_size, channels, height, width]
            if len(pixel_values.shape) != expected_dims:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao validar shapes do batch: {str(e)}")
            return False

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