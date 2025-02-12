import torch
import torch.nn.functional as F
import torch._dynamo
import logging
import os
import time
import gc
import numpy as np
import cv2
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from .base import BaseDetector, BaseCache
import tempfile

logger = logging.getLogger(__name__)

# Configurações globais do PyTorch para otimização em GPU
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch._dynamo.config.suppress_errors = True


class GPUCache(BaseCache):
    """Cache otimizado para GPU."""
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size)
        self.device = torch.device('cuda')


class WeaponDetectorGPU(BaseDetector):
    """Implementação GPU do detector de armas com otimizações para a última versão do OWLv2."""
    
    def __init__(self):
        """Inicializa variáveis básicas."""
        super().__init__()
        self.default_resolution = 640
        self.amp_dtype = torch.float16
        self.preprocess_stream = torch.cuda.Stream()
        self.max_batch_size = 16  # Aumentado para 16
        self.current_batch_size = 8  # Aumentado para 8
        self.min_batch_size = 2
    
    def _initialize(self):
        """Inicializa o modelo e o processador para execução exclusiva em GPU."""
        try:
            # Configurar device
            self.device = self._get_best_device()
            
            # Diretório de cache para o modelo
            cache_dir = os.path.join(tempfile.gettempdir(), 'weapon_detection_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Limpar memória GPU
            self._clear_gpu_memory()
            
            logger.info("Carregando modelo e processador...")
            
            # Carregar processador e modelo com otimizações
            model_name = "google/owlv2-base-patch16"
            self.owlv2_processor = Owlv2Processor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Configurações otimizadas para T4
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=self.amp_dtype,
                device_map="auto",
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Otimizar modelo para inferência
            self.owlv2_model.eval()
            torch.compile(self.owlv2_model)  # Usar torch.compile para otimização
            
            # Usar queries do método base
            self.text_queries = self._get_detection_queries()
            logger.info(f"Total de queries carregadas: {len(self.text_queries)}")
            
            # Processar queries uma única vez com otimização de memória
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                self.processed_text = self.owlv2_processor(
                    text=self.text_queries,
                    return_tensors="pt",
                    padding=True
                )
                
                self.processed_text = {
                    key: val.to(self.device, non_blocking=True) 
                    for key, val in self.processed_text.items()
                }
            
            # Ajustar batch size baseado na memória disponível
            self._adjust_batch_size()
            
            logger.info(f"Inicialização GPU completa! Batch size inicial: {self.current_batch_size}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Erro na inicialização GPU: {str(e)}")
            raise

    def _adjust_batch_size(self):
        """Ajusta o batch size baseado na memória disponível."""
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            free_mem = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            mem_ratio = free_mem / gpu_mem
            
            if mem_ratio < 0.2:  # Menos de 20% livre
                self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            elif mem_ratio > 0.4:  # Mais de 40% livre
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size * 2)
                
            logger.debug(f"Batch size ajustado para {self.current_batch_size} (Memória livre: {mem_ratio:.1%})")
        except Exception as e:
            logger.warning(f"Erro ao ajustar batch size: {str(e)}")
            self.current_batch_size = self.min_batch_size

    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> list:
        """Detecta objetos em uma imagem utilizando a última versão do OWLv2."""
        try:
            self.threshold = threshold
            
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

    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3, resolution: int = 640) -> tuple:
        """Processa um vídeo utilizando GPU com processamento em lote e otimizações para T4."""
        try:
            metrics = {
                "total_time": 0,
                "frame_extraction_time": 0,
                "analysis_time": 0,
                "frames_analyzed": 0,
                "video_duration": 0,
                "device_type": self.device.type,
                "detections": [],
                "technical": {
                    "model": "owlv2-base-patch16",
                    "input_size": f"{resolution}x{resolution}",
                    "threshold": threshold,
                    "batch_size": self.current_batch_size,
                    "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
                }
            }
            
            start_time = time.time()
            frames = self.extract_frames(video_path, fps, resolution)
            metrics["frame_extraction_time"] = time.time() - start_time
            metrics["frames_analyzed"] = len(frames)
            
            if not frames:
                logger.warning("Nenhum frame extraído do vídeo")
                return video_path, metrics
            
            metrics["video_duration"] = len(frames) / (fps or 2)
            analysis_start = time.time()
            
            # Processar frames em lotes com ajuste dinâmico de batch size
            for i in range(0, len(frames), self.current_batch_size):
                try:
                    batch_frames = frames[i:i + self.current_batch_size]
                    
                    # Pré-processamento assíncrono
                    with torch.cuda.stream(self.preprocess_stream):
                        batch_images = [
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            for frame in batch_frames
                        ]
                        
                        batch_inputs = self.owlv2_processor(
                            images=batch_images,
                            return_tensors="pt"
                        )
                        
                        batch_inputs = {
                            key: val.to(self.device, non_blocking=True)
                            for key, val in batch_inputs.items()
                        }
                    
                    # Expandir texto processado para o batch
                    batch_text = {
                        key: val.repeat(len(batch_images), 1) 
                        for key, val in self.processed_text.items()
                    }
                    
                    inputs = {**batch_inputs, **batch_text}
                    
                    # Inferência com mixed precision
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        with torch.no_grad():
                            outputs = self.owlv2_model(**inputs)
                    
                    # Processar resultados
                    target_sizes = torch.tensor([[img.size[::-1] for img in batch_images]], device=self.device)
                    results = self.owlv2_processor.post_process_grounded_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes[0],
                        threshold=threshold
                    )
                    
                    # Verificar detecções
                    for batch_idx, result in enumerate(results):
                        if len(result["scores"]) > 0:
                            frame_idx = i + batch_idx
                            max_score_idx = torch.argmax(result["scores"])
                            score = result["scores"][max_score_idx]
                            
                            if score.item() >= threshold:
                                detection = {
                                    "frame": frame_idx,
                                    "confidence": score.item(),
                                    "box": [int(x) for x in result["boxes"][max_score_idx].tolist()],
                                    "label": self.text_queries[result["labels"][max_score_idx]]
                                }
                                metrics["detections"].append(detection)
                                metrics["analysis_time"] = time.time() - analysis_start
                                metrics["total_time"] = time.time() - start_time
                                return video_path, metrics
                    
                    # Limpar memória e ajustar batch size periodicamente
                    if (i // self.current_batch_size) % 5 == 0:
                        self._clear_gpu_memory()
                        self._adjust_batch_size()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("OOM detectado, reduzindo batch size")
                        self._clear_gpu_memory()
                        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
                        continue
                    raise
            
            metrics["analysis_time"] = time.time() - analysis_start
            metrics["total_time"] = time.time() - start_time
            return video_path, metrics
            
        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            return video_path, metrics

    def _clear_gpu_memory(self):
        """Limpa memória GPU de forma agressiva."""
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            logger.error(f"Erro ao limpar memória GPU: {str(e)}")

    def _get_best_device(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA não está disponível!")
        return torch.device('cuda')

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Pré-processa a imagem com otimizações para GPU."""
        try:
            target_size = (self.default_resolution, self.default_resolution)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            if image.size != target_size:
                ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                
                with torch.cuda.stream(self.preprocess_stream), torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
                    img_tensor = img_tensor.to(self.device, dtype=self.amp_dtype, non_blocking=True)
                    img_tensor = img_tensor / 255.0
                    
                    mode = 'bilinear' if ratio < 1 else 'nearest'
                    img_tensor = F.interpolate(
                        img_tensor,
                        size=new_size,
                        mode=mode,
                        align_corners=False if mode == 'bilinear' else None
                    )

                    if new_size != target_size:
                        final_tensor = torch.zeros(
                            (1, 3, target_size[1], target_size[0]),
                            device=self.device,
                            dtype=self.amp_dtype
                        )
                        pad_left = (target_size[0] - new_size[0]) // 2
                        pad_top = (target_size[1] - new_size[1]) // 2
                        final_tensor[
                            :,
                            :,
                            pad_top:pad_top + new_size[1],
                            pad_left:pad_left + new_size[0]
                        ] = img_tensor

                        img_tensor = final_tensor

                    img_tensor = img_tensor.squeeze(0).permute(1, 2, 0).cpu()
                    image = Image.fromarray((img_tensor.numpy() * 255).astype(np.uint8))

            return image

        except Exception as e:
            logger.error(f"Erro no pré-processamento: {str(e)}")
            return image

    def _get_memory_usage(self):
        """Retorna o uso atual de memória GPU em porcentagem."""
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated + reserved) / total * 100
        except Exception as e:
            logger.error(f"Erro ao obter uso de memória GPU: {str(e)}")
            return 0

    def _apply_nms(self, detections: list, iou_threshold: float = 0.5) -> list:
        """Aplica Non-Maximum Suppression nas detecções usando operações em GPU."""
        try:
            if not detections:
                return []

            # Converter detecções para tensores na GPU
            boxes = torch.tensor([[d["box"][0], d["box"][1], d["box"][2], d["box"][3]] for d in detections], device=self.device)
            scores = torch.tensor([d["confidence"] for d in detections], device=self.device)
            labels = [d["label"] for d in detections]

            # Calcular áreas dos boxes
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
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
                xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
                yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
                xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
                yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])

                w = torch.clamp(xx2 - xx1, min=0)
                h = torch.clamp(yy2 - yy1, min=0)
                inter = w * h

                # Calcular IoU
                ovr = inter / (area[i] + area[order[1:]] - inter)
                
                # Encontrar boxes com IoU menor que o threshold
                ids = (ovr <= iou_threshold).nonzero().squeeze()
                if ids.numel() == 0:
                    break
                order = order[ids + 1]

            # Construir lista de detecções filtradas
            filtered_detections = []
            for idx in keep:
                filtered_detections.append({
                    "confidence": scores[idx].item(),
                    "box": boxes[idx].tolist(),
                    "label": labels[idx]
                })

            return filtered_detections

        except Exception as e:
            logger.error(f"Erro ao aplicar NMS na GPU: {str(e)}")
            return []

    def _should_clear_cache(self):
        """Determina se o cache deve ser limpo baseado no uso de memória."""
        try:
            memory_usage = self._get_memory_usage()
            if memory_usage > 90:
                return True
            if memory_usage > 75 and not hasattr(self, '_last_cache_clear'):
                return True
            if hasattr(self, '_last_cache_clear'):
                time_since_last_clear = time.time() - self._last_cache_clear
                if memory_usage > 80 and time_since_last_clear > 300:
                    return True
            return False
        except Exception as e:
            logger.error(f"Erro ao verificar necessidade de limpeza: {str(e)}")
            return False

    def clear_cache(self):
        """Limpa o cache de resultados e libera memória quando necessário."""
        try:
            if self._should_clear_cache():
                if hasattr(self, 'result_cache'):
                    self.result_cache.clear()
                torch.cuda.empty_cache()
                gc.collect()
                self._last_cache_clear = time.time()
                logger.info(f"Cache GPU limpo com sucesso. Uso de memória: {self._get_memory_usage():.1f}%")
            else:
                logger.debug("Limpeza de cache não necessária no momento")
        except Exception as e:
            logger.error(f"Erro ao limpar cache GPU: {str(e)}")