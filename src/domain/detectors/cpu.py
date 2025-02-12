import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional, Union
import os
from tqdm import tqdm
import json
from pathlib import Path
from contextlib import nullcontext
import threading
import hashlib
import pickle
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import subprocess
import shutil
import traceback
import psutil
import logging
import gc
import sys
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from .base import BaseDetector, BaseCache


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

class CPUCache(BaseCache):
    """Cache otimizado para CPU."""
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size)
        self.device = torch.device('cpu')

class WeaponDetectorCPU(BaseDetector):
    """Implementação CPU do detector de armas."""
    def __init__(self):
        """Inicializa variáveis básicas."""
        super().__init__()
        self.default_resolution = 640
        self.device = torch.device('cpu')

    def _get_best_device(self):
        return torch.device('cpu')

    def _initialize(self):
        """Inicializa o modelo e o processador para execução em CPU."""
        try:
            # Configurações otimizadas para CPU
            torch.set_num_threads(min(8, os.cpu_count()))
            torch.set_num_interop_threads(min(8, os.cpu_count()))
            
            # Carregar modelo com configurações otimizadas
            cache_dir = os.path.join(tempfile.gettempdir(), 'weapon_detection_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            model_name = "google/owlv2-base-patch16"
            logger.info("Carregando modelo e processador...")
            
            self.owlv2_processor = Owlv2Processor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )

            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)

            self.owlv2_model.eval()

            # Usar queries do método base
            self.text_queries = self._get_detection_queries()
            logger.info(f"Total de queries carregadas: {len(self.text_queries)}")
            
            # Processar queries uma única vez
            logger.info("Processando queries...")
            self.processed_text = self.owlv2_processor(
                text=self.text_queries,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Inicializar cache
            cache_size = int(os.getenv('RESULT_CACHE_SIZE', '1000'))
            self.result_cache = CPUCache(max_size=cache_size)

            logger.info("Inicialização CPU completa!")
            self._initialized = True

        except Exception as e:
            logger.error(f"Erro na inicialização CPU: {str(e)}")
            raise

    def _apply_nms(self, detections: list, iou_threshold: float = 0.5) -> list:
        """Aplica NMS usando operações em CPU."""
        try:
            if not detections:
                return []

            boxes = torch.tensor([[d["box"][0], d["box"][1], d["box"][2], d["box"][3]] for d in detections])
            scores = torch.tensor([d["confidence"] for d in detections])
            labels = [d["label"] for d in detections]

            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            _, order = scores.sort(descending=True)

            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                i = order[0]
                keep.append(i.item())

                xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
                yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
                xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
                yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])

                w = torch.clamp(xx2 - xx1, min=0)
                h = torch.clamp(yy2 - yy1, min=0)
                inter = w * h

                ovr = inter / (area[i] + area[order[1:]] - inter)
                ids = (ovr <= iou_threshold).nonzero().squeeze()
                if ids.numel() == 0:
                    break
                order = order[ids + 1]

            filtered_detections = []
            for idx in keep:
                filtered_detections.append({
                    "confidence": scores[idx].item(),
                    "box": boxes[idx].tolist(),
                    "label": labels[idx]
                })
            return filtered_detections

        except Exception as e:
            logger.error(f"Erro ao aplicar NMS: {str(e)}")
            return []

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Pré-processa a imagem para o tamanho 640x640 e garante RGB."""
        try:
            target_size = (640, 640)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if image.size != target_size:
                ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
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

    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> list:
        """Detecta objetos em uma imagem utilizando CPU."""
        try:
            image = self._preprocess_image(image)
            with torch.no_grad():
                image_inputs = self.owlv2_processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                inputs = {**image_inputs, **self.processed_text}
                outputs = self.owlv2_model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]])
                results = self.owlv2_processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=threshold
                )[0]

                detections = []
                for score, box, label in zip(results["scores"], results["boxes"], results["labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append({
                        "confidence": score.item(),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "label": self.text_queries[label]
                    })
                return self._apply_nms(detections)

        except Exception as e:
            logger.error(f"Erro em detect_objects: {str(e)}")
            return []

    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3, resolution: int = 640) -> tuple:
        """Processa um vídeo utilizando CPU. Para na primeira detecção encontrada."""
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
                    "model": "owlv2-base-patch16-ensemble",
                    "input_size": f"{resolution}x{resolution}",
                    "nms_threshold": 0.5,
                    "preprocessing": "basic",
                    "early_stop": True
                },
            }

            start_time = time.time()
            t0 = time.time()
            frames = self.extract_frames(video_path, fps, resolution)
            metrics["frame_extraction_time"] = time.time() - t0
            metrics["frames_analyzed"] = len(frames)

            if not frames:
                logger.warning("Nenhum frame extraído do vídeo")
                return video_path, metrics

            metrics["video_duration"] = len(frames) / (fps or 2)
            t0 = time.time()
            detections = []
            frames_processed = 0

            # Processar um frame por vez para otimizar memória e permitir parada precoce
            for frame_idx, frame in enumerate(frames):
                frames_processed += 1
                
                # Converter frame para RGB e pré-processar
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image = self._preprocess_image(image)

                # Detectar objetos com threshold direto
                with torch.no_grad():
                    image_inputs = self.owlv2_processor(
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    inputs = {**image_inputs, **self.processed_text}
                    outputs = self.owlv2_model(**inputs)

                    target_sizes = torch.tensor([image.size[::-1]])
                    results = self.owlv2_processor.post_process_grounded_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=threshold  # Aplicar threshold diretamente
                    )[0]

                    # Se encontrou alguma detecção acima do threshold
                    if len(results["scores"]) > 0:
                        # Pegar a detecção com maior confiança
                        max_score_idx = torch.argmax(results["scores"])
                        score = results["scores"][max_score_idx].item()
                        box = results["boxes"][max_score_idx].tolist()
                        label = results["labels"][max_score_idx].item()

                        detections.append({
                            "frame": frame_idx,
                            "confidence": score,
                            "box": [int(x) for x in box],
                            "label": self.text_queries[label]
                        })

                        # Atualizar métricas e parar o processamento
                        metrics["frames_processed_until_detection"] = frames_processed
                        metrics["analysis_time"] = time.time() - t0
                        metrics["total_time"] = time.time() - start_time
                        metrics["detections"] = detections
                        logger.info(f"Detecção encontrada após processar {frames_processed} frames")
                        return video_path, metrics

                # Liberar memória a cada 10 frames
                if frames_processed % 10 == 0:
                    gc.collect()

            # Se chegou aqui, não encontrou nenhuma detecção
            metrics["analysis_time"] = time.time() - t0
            metrics["total_time"] = time.time() - start_time
            metrics["frames_processed_until_detection"] = frames_processed
            metrics["detections"] = detections
            return video_path, metrics

        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            return video_path, {}

    def extract_frames(self, video_path: str, fps: int = 2, resolution: int = 480) -> list:
        """Extrai frames de um vídeo utilizando ffmpeg."""
        frames = []
        temp_dir = Path(tempfile.mkdtemp())
        try:
            threads = min(os.cpu_count(), 4)  # Menor número de threads para CPU
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
            chunk_size = 50  # Menor chunk size para CPU
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

    def clear_cache(self):
        """Limpa o cache de resultados e libera memória."""
        try:
            if hasattr(self, 'result_cache'):
                self.result_cache.clear()
            gc.collect()
            logger.info("Cache CPU limpo com sucesso")
        except Exception as e:
            logger.error(f"Erro ao limpar cache CPU: {str(e)}")

    def _apply_nms(self, detections: list, iou_threshold: float = 0.5) -> list:
        """Aplica NMS usando operações em CPU."""
        try:
            if not detections:
                return []

            boxes = torch.tensor([[d["box"][0], d["box"][1], d["box"][2], d["box"][3]] for d in detections])
            scores = torch.tensor([d["confidence"] for d in detections])
            labels = [d["label"] for d in detections]

            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            _, order = scores.sort(descending=True)

            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                i = order[0]
                keep.append(i.item())

                xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
                yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
                xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
                yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])

                w = torch.clamp(xx2 - xx1, min=0)
                h = torch.clamp(yy2 - yy1, min=0)
                inter = w * h

                ovr = inter / (area[i] + area[order[1:]] - inter)
                ids = (ovr <= iou_threshold).nonzero().squeeze()
                if ids.numel() == 0:
                    break
                order = order[ids + 1]

            filtered_detections = []
            for idx in keep:
                filtered_detections.append({
                    "confidence": scores[idx].item(),
                    "box": boxes[idx].tolist(),
                    "label": labels[idx]
                })
            return filtered_detections

        except Exception as e:
            logger.error(f"Erro ao aplicar NMS: {str(e)}")
            return []

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Pré-processa a imagem para o tamanho 640x640 e garante RGB."""
        try:
            target_size = (640, 640)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if image.size != target_size:
                ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
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

    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> list:
        """Detecta objetos em uma imagem utilizando CPU."""
        try:
            image = self._preprocess_image(image)
            with torch.no_grad():
                image_inputs = self.owlv2_processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                inputs = {**image_inputs, **self.processed_text}
                outputs = self.owlv2_model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]])
                results = self.owlv2_processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=threshold
                )[0]

                detections = []
                for score, box, label in zip(results["scores"], results["boxes"], results["labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append({
                        "confidence": score.item(),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "label": self.text_queries[label]
                    })
                return self._apply_nms(detections)

        except Exception as e:
            logger.error(f"Erro em detect_objects: {str(e)}")
            return []

    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3, resolution: int = 640) -> tuple:
        """Processa um vídeo utilizando CPU. Para na primeira detecção encontrada."""
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
                    "model": "owlv2-base-patch16-ensemble",
                    "input_size": f"{resolution}x{resolution}",
                    "nms_threshold": 0.5,
                    "preprocessing": "basic",
                    "early_stop": True
                },
            }

            start_time = time.time()
            t0 = time.time()
            frames = self.extract_frames(video_path, fps, resolution)
            metrics["frame_extraction_time"] = time.time() - t0
            metrics["frames_analyzed"] = len(frames)

            if not frames:
                logger.warning("Nenhum frame extraído do vídeo")
                return video_path, metrics

            metrics["video_duration"] = len(frames) / (fps or 2)
            t0 = time.time()
            detections = []
            frames_processed = 0

            # Processar um frame por vez para otimizar memória e permitir parada precoce
            for frame_idx, frame in enumerate(frames):
                frames_processed += 1
                
                # Converter frame para RGB e pré-processar
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image = self._preprocess_image(image)

                # Detectar objetos com threshold direto
                with torch.no_grad():
                    image_inputs = self.owlv2_processor(
                        images=image,
                        return_tensors="pt"
                    ).to(self.device)
                    inputs = {**image_inputs, **self.processed_text}
                    outputs = self.owlv2_model(**inputs)

                    target_sizes = torch.tensor([image.size[::-1]])
                    results = self.owlv2_processor.post_process_grounded_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=threshold  # Aplicar threshold diretamente
                    )[0]

                    # Se encontrou alguma detecção acima do threshold
                    if len(results["scores"]) > 0:
                        # Pegar a detecção com maior confiança
                        max_score_idx = torch.argmax(results["scores"])
                        score = results["scores"][max_score_idx].item()
                        box = results["boxes"][max_score_idx].tolist()
                        label = results["labels"][max_score_idx].item()

                        detections.append({
                            "frame": frame_idx,
                            "confidence": score,
                            "box": [int(x) for x in box],
                            "label": self.text_queries[label]
                        })

                        # Atualizar métricas e parar o processamento
                        metrics["frames_processed_until_detection"] = frames_processed
                        metrics["analysis_time"] = time.time() - t0
                        metrics["total_time"] = time.time() - start_time
                        metrics["detections"] = detections
                        logger.info(f"Detecção encontrada após processar {frames_processed} frames")
                        return video_path, metrics

                # Liberar memória a cada 10 frames
                if frames_processed % 10 == 0:
                    gc.collect()

            # Se chegou aqui, não encontrou nenhuma detecção
            metrics["analysis_time"] = time.time() - t0
            metrics["total_time"] = time.time() - start_time
            metrics["frames_processed_until_detection"] = frames_processed
            metrics["detections"] = detections
            return video_path, metrics

        except Exception as e:
            logger.error(f"Erro ao processar vídeo: {str(e)}")
            return video_path, {}

    def extract_frames(self, video_path: str, fps: int = 2, resolution: int = 480) -> list:
        """Extrai frames de um vídeo utilizando ffmpeg."""
        frames = []
        temp_dir = Path(tempfile.mkdtemp())
        try:
            threads = min(os.cpu_count(), 4)  # Menor número de threads para CPU
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
            chunk_size = 50  # Menor chunk size para CPU
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

    def clear_cache(self):
        """Limpa cache e libera memória."""
        self.result_cache.clear()
        gc.collect() 