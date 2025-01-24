import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional, Union
import os
from tqdm import tqdm
import json
from pathlib import Path
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
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

# Carregar variáveis de ambiente
load_dotenv()

class WeaponDetectorSingleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WeaponDetectorSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not WeaponDetectorSingleton._initialized:
            self._initialize()
            WeaponDetectorSingleton._initialized = True
    
    def _initialize(self):
        """Initialize the model and processor only once."""
        print("Initializing WeaponDetector Singleton...")
        
        # Configurar device
        self.device = self._get_best_device()
        print(f"Using device: {self.device}")
        
        # Configurar otimizações do PyTorch
        torch.backends.cudnn.benchmark = True
        if self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Configurar diretórios de cache
        self.cache_dir = Path("./.cache")
        self.model_cache_dir = self.cache_dir / "model"
        self.video_cache_dir = self.cache_dir / "videos"
        
        # Criar diretórios de cache
        self.cache_dir.mkdir(exist_ok=True)
        self.model_cache_dir.mkdir(exist_ok=True)
        self.video_cache_dir.mkdir(exist_ok=True)
        
        # Carregar modelo e processador (apenas uma vez)
        print("Loading model and processor...")
        self.owlv2_processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch16",  
            cache_dir=str(self.model_cache_dir)
        )
        self.owlv2_model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch16",  
            cache_dir=str(self.model_cache_dir)
        ).to(self.device)
        
        # Otimizar modelo para inferência
        self.owlv2_model.eval()
        if self.device.type == 'cuda':
            self.owlv2_model = torch.compile(self.owlv2_model)
        
        # Inicializar thread pool para processamento paralelo
        self.thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # Definir queries de detecção (apenas uma vez)
        self.dangerous_objects = self._get_detection_queries()
        
        # Pre-processar queries (apenas uma vez)
        print("Pre-processing detection queries...")
        text_queries = self.dangerous_objects
        
        # Processar todas as queries de uma vez
        self.text_inputs = self.owlv2_processor(
            text=text_queries,
            return_tensors="pt",
            padding=True
        ).to(self.device)
    
    def _get_best_device(self) -> torch.device:
        """Get the best available device for computation."""
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            torch.set_num_threads(os.cpu_count())
            return torch.device('cpu')
    
    def _get_detection_queries(self) -> List[str]:
        """Retorna as queries para detecção de objetos perigosos."""
        # Definir categorias de objetos perigosos
        edged_weapons = [
            "knife", "blade", "dagger", "machete", "sword",
            "combat knife", "hunting knife", "kitchen knife", "pocket knife",
            "switchblade", "butterfly knife", "folding knife", "fixed blade knife",
            "tactical knife", "utility knife", "box cutter", "razor blade"
        ]
        
        sharp_objects = [
            "sharp object", "pointed object", "cutting tool",
            "scissors", "shears", "glass shard", "broken glass",
            "screwdriver", "ice pick", "awl", "needle", "metal spike",
            "sharpened object", "blade weapon"
        ]
        
        firearms = [
            "gun", "pistol", "rifle", "firearm", "handgun",
            "revolver", "shotgun", "assault rifle", "weapon"
        ]
        
        # Contextos visuais importantes
        visual_contexts = [
            "close-up", "clear view", "detailed photo",
            "high resolution image", "focused shot",
            "visible", "clear photo", "sharp image"
        ]
        
        # Características de perigo
        danger_contexts = [
            "dangerous", "threatening", "weapon",
            "harmful", "hazardous", "lethal",
            "menacing", "unsafe"
        ]
        
        # Características físicas
        physical_traits = [
            "metallic", "sharp", "pointed",
            "steel", "shiny", "reflective",
            "blade-like", "edged"
        ]
        
        queries = []
        
        # Gerar queries para armas brancas
        for weapon in edged_weapons:
            base_queries = [
                f"a {weapon} with sharp edge",
                f"a dangerous {weapon}",
                f"{weapon} blade visible",
                f"clear photo of {weapon}",
                f"metallic {weapon}",
                f"{weapon} weapon"
            ]
            queries.extend(base_queries)
            
            # Adicionar variações com contextos
            for context in visual_contexts:
                queries.append(f"a {context} of a {weapon}")
            
            for trait in physical_traits:
                queries.append(f"a {trait} {weapon}")
        
        # Gerar queries para objetos cortantes
        for obj in sharp_objects:
            base_queries = [
                f"a sharp {obj}",
                f"dangerous {obj}",
                f"pointed {obj}",
                f"clear view of {obj}"
            ]
            queries.extend(base_queries)
            
            # Adicionar variações com contextos de perigo
            for context in danger_contexts:
                queries.append(f"a {context} {obj}")
        
        # Gerar queries para armas de fogo
        for weapon in firearms:
            base_queries = [
                f"a {weapon} visible",
                f"clear photo of {weapon}",
                f"dangerous {weapon}",
                f"{weapon} in frame"
            ]
            queries.extend(base_queries)
            
            # Adicionar variações com contextos
            for context in visual_contexts[:3]:  # Limitar para não ter muitas queries
                queries.append(f"a {context} of a {weapon}")
        
        # Remover duplicatas e limitar número de queries
        queries = list(set(queries))[:200]  # Limitar a 200 queries mais relevantes
        
        return queries
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Pré-processa a imagem para melhor detecção com OWL-ViT patch16."""
        try:
            # Converter para RGB se necessário
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 1. Redimensionamento otimizado para patch16
            target_size = (1280, 1280)  # Aumentado para patch16 (múltiplo de 16)
            
            # Calcular novo tamanho mantendo aspect ratio
            ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
            new_size = tuple(int(dim * ratio) for dim in image.size)
            
            # Ajustar para múltiplo de 16 (patch16)
            new_size = tuple(((dim + 15) // 16) * 16 for dim in new_size)
            
            # Redimensionar com Lanczos
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 2. Converter para array numpy para processamento avançado
            img_array = np.array(image)
            
            # 3. Denoising bilateral mais suave (patch16 é mais sensível a detalhes)
            denoised = cv2.bilateralFilter(img_array, d=7, sigmaColor=50, sigmaSpace=50)
            
            # 4. Ajuste de contraste adaptativo mais suave
            lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE mais suave para preservar detalhes
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
            l = clahe.apply(l)
            
            # Recombinar canais
            lab = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 5. Sharpening mais suave
            kernel = np.array([[-0.5,-0.5,-0.5],
                             [-0.5,  5,-0.5],
                             [-0.5,-0.5,-0.5]]) / 2
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 6. Normalização de cores
            normalized = np.zeros(sharpened.shape, sharpened.dtype)
            normalized = cv2.normalize(sharpened, normalized, 0, 255, cv2.NORM_MINMAX)
            
            # 7. Ajuste de gama mais suave
            gamma = 1.1  # Mais suave para preservar detalhes
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            gamma_corrected = cv2.LUT(normalized, table)
            
            # 8. Aumentar saturação levemente
            hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2HSV)
            hsv[:,:,1] = hsv[:,:,1] * 1.2  # 20% de aumento
            hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
            saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # 9. Criar padding com borda refletida
            # Calcular padding para manter múltiplo de 16
            h, w = saturated.shape[:2]
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            
            padded = cv2.copyMakeBorder(
                saturated,
                pad_h//2, (pad_h+1)//2,
                pad_w//2, (pad_w+1)//2,
                cv2.BORDER_REFLECT
            )
            
            # 10. Redimensionar para o tamanho final com padding preto
            final_image = Image.new('RGB', target_size, (0, 0, 0))
            padded_pil = Image.fromarray(padded)
            paste_pos = ((target_size[0] - padded.shape[1]) // 2,
                        (target_size[1] - padded.shape[0]) // 2)
            final_image.paste(padded_pil, paste_pos)
            
            return final_image
            
        except Exception as e:
            print(f"Erro no pré-processamento: {str(e)}")
            return image

    @torch.inference_mode()
    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> List[Dict]:
        """Detect objects in image using pre-processed queries."""
        try:
            # Melhorar qualidade da imagem
            image = self._preprocess_image(image)
            
            # Processar imagem com diferentes escalas (mais granular para patch16)
            scales = [0.85, 1.0, 1.15]  # Escalas mais próximas para patch16
            all_detections = []
            
            for scale in scales:
                # Redimensionar imagem (mantendo múltiplo de 16)
                w, h = image.size
                scaled_size = (int(w * scale), int(h * scale))
                scaled_size = tuple(((dim + 15) // 16) * 16 for dim in scaled_size)
                scaled_image = image.resize(scaled_size, Image.Resampling.LANCZOS)
                
                # Processar imagem
                image_inputs = self.owlv2_processor(
                    images=scaled_image,
                    return_tensors="pt"
                ).to(self.device)
                
                # Combinar com as queries pré-processadas
                inputs = {**image_inputs, **self.text_inputs}
                
                # Usar autocast para melhor performance
                with torch.cuda.amp.autocast() if self.device.type == 'cuda' else nullcontext():
                    outputs = self.owlv2_model(**inputs)
                
                # Processar resultados
                target_sizes = torch.tensor([scaled_image.size[::-1]], device=self.device)
                results = self.owlv2_processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=threshold
                )[0]
                
                # Ajustar coordenadas para escala original
                scores = results["scores"]
                boxes = results["boxes"]
                labels = results["labels"]
                
                for score, box, label in zip(scores, boxes, labels):
                    # Converter box para escala original
                    x1, y1, x2, y2 = box.tolist()
                    orig_box = [
                        int(x1 / scale),
                        int(y1 / scale),
                        int(x2 / scale),
                        int(y2 / scale)
                    ]
                    
                    all_detections.append({
                        "score": score.item(),
                        "box": orig_box,
                        "label": self.dangerous_objects[label]
                    })
            
            # Non-maximum suppression com IoU mais baixo para patch16
            filtered_detections = []
            all_detections.sort(key=lambda x: x["score"], reverse=True)
            
            while all_detections:
                best = all_detections.pop(0)
                filtered_detections.append(best)
                
                # Remover detecções sobrepostas (IoU mais baixo para patch16)
                all_detections = [
                    d for d in all_detections
                    if self._calculate_iou(best["box"], d["box"]) < 0.4  # IoU mais baixo
                ]
            
            return filtered_detections
            
        except Exception as e:
            print(f"Erro em detect_objects: {str(e)}")
            return []
            
    def _calculate_iou(self, box1, box2):
        """Calcula IoU (Intersection over Union) entre duas boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _get_cache_key(self, video_path: str, fps: int, threshold: float) -> str:
        """Generate cache key based on video content and parameters."""
        hasher = hashlib.sha256()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        params = f"{fps}_{threshold}"
        hasher.update(params.encode())
        return hasher.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.video_cache_dir / f"{cache_key}.cache"
    
    def extract_frames(self, video_path: str, fps: int = 2) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video using ffmpeg."""
        frames = []
        
        # Criar diretório temporário para os frames
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Comando ffmpeg para extrair frames
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', f'fps={fps},scale=768:768:force_original_aspect_ratio=decrease,pad=768:768:(ow-iw)/2:(oh-ih)/2',
                '-frame_pts', '1',  # Adiciona timestamp nos frames
                f'{temp_dir}/%d.jpg'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Ler frames extraídos
            frame_files = sorted(temp_dir.glob('*.jpg'), key=lambda x: int(x.stem))
            
            for frame_file in tqdm(frame_files, desc="Carregando frames"):
                # Ler frame
                frame = cv2.imread(str(frame_file))
                
                # Calcular timestamp baseado no número do frame
                frame_number = int(frame_file.stem)
                timestamp = (frame_number - 1) / fps
                
                frames.append((timestamp, frame))
                
        finally:
            # Limpar diretório temporário
            shutil.rmtree(temp_dir)
            
        return frames

    def analyze_video(self, video_path: str, threshold: float = 0.3, fps: int = 5, cancel_event=None) -> Tuple[List[Dict], Dict, Dict]:
        """Analyze video for dangerous objects."""
        metrics = {
            "total_time": 0,
            "frame_extraction_time": 0,
            "analysis_time": 0,
            "frames_analyzed": 0,
            "video_duration": 0,
            "device_type": self.device.type
        }
        
        start_time = time.time()
        
        # Extrair frames do vídeo
        t0 = time.time()
        frames = self.extract_frames(video_path, fps)
        metrics["frame_extraction_time"] = time.time() - t0
        metrics["frames_analyzed"] = len(frames)
        
        if not frames:
            print("No frames extracted from video")
            return [], {"risk_status": "unknown", "time_ranges": []}, metrics
        
        # Calcular duração do vídeo baseado no último timestamp
        metrics["video_duration"] = max(frames[-1][0] if frames else 1.0, 1.0)  # Mínimo de 1s para evitar divisão por zero
        
        t0 = time.time()
        detections = []
        timestamps = set()
        
        # Processar frames em lotes
        batch_size = 4 if self.device.type == 'cuda' else 2
        for i in range(0, len(frames), batch_size):
            if cancel_event and cancel_event.is_set():
                return [], {}, metrics
            
            batch_frames = frames[i:i + batch_size]
            futures = []
            
            # Submeter frames para detecção em paralelo
            for timestamp, frame in batch_frames:
                # Converter BGR para RGB e para PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                future = self.thread_pool.submit(
                    self.detect_objects,
                    frame_pil,
                    threshold
                )
                futures.append((future, timestamp, frame))
            
            # Coletar resultados
            for future, timestamp, original_frame in futures:
                try:
                    frame_detections = future.result()
                    timestamps.add(timestamp)
                    
                    # Adicionar informações extras às detecções
                    for detection in frame_detections:
                        detection.update({
                            "timestamp": timestamp,
                            "frame": original_frame,  # Manter como numpy array
                            "type": detection["label"]
                        })
                    
                    detections.extend(frame_detections)
                except Exception as e:
                    print(f"Error processing frame at {timestamp}: {str(e)}")
        
        metrics["analysis_time"] = time.time() - t0
        metrics["total_time"] = time.time() - start_time
        
        # Gerar resumo
        json_summary = {
            "risk_status": "danger" if detections else "safe",
            "time_ranges": self._create_time_ranges(sorted(timestamps))
        }
        
        # Gerar vídeo com análise
        if detections:
            output_path = self.generate_output_video(video_path, detections, fps=fps)
            json_summary["output_video"] = output_path
        
        return detections, json_summary, metrics

    def generate_output_video(self, video_path: str, detections: List[Dict], output_path: str = None, fps: int = 30) -> str:
        """Gera vídeo final usando apenas os frames analisados."""
        try:
            if not output_path:
                output_path = f"{os.path.splitext(video_path)[0]}_analyzed.mp4"
            
            # Obter FPS do vídeo original
            cap = cv2.VideoCapture(video_path)
            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            # Gerar filtros para as detecções
            draw_filter = self.generate_ffmpeg_filter(detections, video_fps)
            if not draw_filter:
                print("Nenhuma detecção para desenhar")
                return video_path
            
            # Construir comando ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"format=rgb24,{draw_filter},format=yuv420p",
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-c:a', 'copy',
                '-movflags', '+faststart',
                output_path
            ]
            
            print("Comando ffmpeg:", " ".join(cmd))
            
            # Executar comando com timeout
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Esperar com timeout
                stdout, stderr = process.communicate(timeout=300)  # 5 minutos
                
                if process.returncode != 0:
                    print(f"Erro ao gerar vídeo: {stderr.decode()}")
                    return video_path
                
                return output_path
                
            except subprocess.TimeoutExpired:
                process.kill()
                print("Timeout ao gerar vídeo")
                return video_path
                
        except Exception as e:
            print(f"Erro ao gerar vídeo: {str(e)}")
            traceback.print_exc()
            return video_path

    def draw_bounding_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Desenha os bounding boxes nos frames."""
        try:
            # Converter para PIL para desenho
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            
            # Configurações visuais
            box_color = (255, 0, 0)  # Vermelho para risco
            text_color = (255, 255, 255)  # Branco para texto
            font_size = max(1, int(image.size[0] / 40))  # Tamanho proporcional à imagem
            
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Desenhar cada detecção
            for detection in detections:
                # Extrair coordenadas
                box = detection["box"]
                x1, y1, x2, y2 = map(int, box)
                
                # Desenhar retângulo
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                
                # Preparar texto
                text = "RISK"
                text_w, text_h = draw.textsize(text, font=font)
                
                # Desenhar fundo para o texto
                margin = 5
                text_box = [x1, y1 - text_h - 2*margin, x1 + text_w + 2*margin, y1]
                draw.rectangle(text_box, fill=box_color)
                
                # Desenhar texto
                text_pos = (x1 + margin, y1 - text_h - margin)
                draw.text(text_pos, text, fill=text_color, font=font)
            
            return np.array(image)
            
        except Exception as e:
            print(f"Erro ao desenhar bounding boxes: {str(e)}")
            return frame

    def generate_ffmpeg_filter(self, detections, fps):
        """
        Gera um filtro ffmpeg para desenhar bounding boxes no vídeo.
        """
        try:
            if not detections:
                return ""
            
            # Agrupar detecções por timestamp
            detections_by_time = {}
            for d in detections:
                t = d.get("timestamp", 0)
                if t not in detections_by_time:
                    detections_by_time[t] = []
                detections_by_time[t].append(d)
            
            # Gerar comandos de desenho
            draw_commands = []
            
            for timestamp, frame_detections in detections_by_time.items():
                frame_number = int(timestamp * fps)
                
                for detection in frame_detections:
                    box = detection["box"]
                    score = detection["score"]
                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Comando para o retângulo vermelho
                    rect_cmd = f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color=red:thickness=3"
                    
                    # Texto "RISK {score}%"
                    score_percent = round(score * 100, 1)
                    text = f"RISK {score_percent}%"
                    text_y = max(30, y1 - 10)  # Evitar texto fora da tela
                    
                    # Calcular largura do texto (aproximada)
                    text_width = len(text) * 10  # Aproximadamente 10 pixels por caractere
                    
                    # Fundo vermelho para o texto
                    text_bg = f"drawbox=x={x1}:y={text_y-20}:w={text_width}:h=20:color=red:thickness=fill"
                    
                    # Texto em branco com score
                    text_cmd = f"drawtext=text='{text}':x={x1+5}:y={text_y-15}:fontsize=16:fontcolor=white:shadowcolor=black:shadowx=1:shadowy=1"
                    
                    # Adicionar enable condition para o frame específico
                    for cmd in [rect_cmd, text_bg, text_cmd]:
                        draw_commands.append(f"{cmd}:enable='eq(n,{frame_number})'")
            
            # Combinar todos os comandos
            if draw_commands:
                return ",".join(draw_commands)
            
            return ""
            
        except Exception as e:
            print(f"Erro ao gerar filtro ffmpeg: {str(e)}")
            return ""

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load results from cache if available and valid."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                # Cache é válido por 24 horas
                cache_time = datetime.fromisoformat(cache_data["timestamp"])
                if (datetime.now() - cache_time).total_seconds() < 86400:
                    return cache_data["data"]
            except Exception as e:
                print(f"Erro ao carregar cache: {str(e)}")
                # Remover cache corrompido
                cache_path.unlink(missing_ok=True)
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save results to cache with metadata."""
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _create_time_ranges(self, timestamps: List[float]) -> List[Dict]:
        """Create time ranges with optimized gap threshold."""
        timestamps_list = sorted(list(timestamps))
        ranges = []
        if timestamps_list:
            range_start = timestamps_list[0]
            prev_time = timestamps_list[0]
            
            for t in timestamps_list[1:]:
                if t - prev_time > 0.5:
                    ranges.append((range_start, prev_time))
                    range_start = t
                prev_time = t
            ranges.append((range_start, prev_time))
        
        return [
            {
                "start": start,
                "end": end,
                "duration": end - start
            }
            for start, end in ranges
        ]

    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3) -> Tuple[str, Dict]:
        """Process video and return path to analyzed video and technical details."""
        try:
            start_time = time.time()
            metrics = {
                "performance": {
                    "total_time": 0,
                    "fps_processing": 0,
                    "frames_processed": 0,
                    "avg_detection_time": 0,
                    "avg_preprocessing_time": 0
                },
                "detection": {
                    "total_detections": 0,
                    "detections_by_scale": {},
                    "confidence_distribution": {
                        "0.3-0.4": 0, "0.4-0.5": 0,
                        "0.5-0.6": 0, "0.6-0.7": 0,
                        "0.7-0.8": 0, "0.8-0.9": 0,
                        "0.9-1.0": 0
                    },
                    "detection_sizes": {
                        "small": 0,  # < 32x32
                        "medium": 0, # 32x32 - 96x96
                        "large": 0   # > 96x96
                    },
                    "false_positives_filtered": 0
                },
                "preprocessing": {
                    "avg_image_size": {"width": 0, "height": 0},
                    "resize_stats": {"min": 0, "max": 0, "avg": 0},
                    "brightness_stats": {"min": 0, "max": 0, "avg": 0},
                    "contrast_stats": {"min": 0, "max": 0, "avg": 0}
                },
                "memory": {
                    "peak_memory_mb": 0,
                    "avg_memory_mb": 0
                },
                "technical": {
                    "model": "owlv2-base-patch16",
                    "input_size": "1280x1280",
                    "scales": [0.85, 1.0, 1.15],
                    "nms_threshold": 0.4,
                    "preprocessing_steps": [
                        "bilateral_filter(d=7,σ=50)",
                        "clahe(limit=2.0,grid=16x16)",
                        "adaptive_sharpening",
                        "gamma_correction(1.1)",
                        "saturation_boost(1.2)"
                    ]
                }
            }
            
            # Processar frames
            total_preprocessing_time = 0
            total_detection_time = 0
            memory_samples = []
            all_detections = []
            
            frames = self.extract_frames(video_path, fps)
            metrics["performance"]["frames_processed"] = len(frames)
            
            for timestamp, frame in frames:
                frame_start = time.time()
                
                # Métricas de pré-processamento
                preprocess_start = time.time()
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                orig_size = pil_image.size
                metrics["preprocessing"]["avg_image_size"]["width"] += orig_size[0]
                metrics["preprocessing"]["avg_image_size"]["height"] += orig_size[1]
                
                # Análise de brilho e contraste
                img_array = np.array(pil_image)
                brightness = np.mean(img_array)
                contrast = np.std(img_array)
                
                metrics["preprocessing"]["brightness_stats"]["min"] = min(metrics["preprocessing"]["brightness_stats"]["min"], brightness) if metrics["preprocessing"]["brightness_stats"]["min"] else brightness
                metrics["preprocessing"]["brightness_stats"]["max"] = max(metrics["preprocessing"]["brightness_stats"]["max"], brightness)
                metrics["preprocessing"]["brightness_stats"]["avg"] += brightness
                
                metrics["preprocessing"]["contrast_stats"]["min"] = min(metrics["preprocessing"]["contrast_stats"]["min"], contrast) if metrics["preprocessing"]["contrast_stats"]["min"] else contrast
                metrics["preprocessing"]["contrast_stats"]["max"] = max(metrics["preprocessing"]["contrast_stats"]["max"], contrast)
                metrics["preprocessing"]["contrast_stats"]["avg"] += contrast
                
                processed_image = self._preprocess_image(pil_image)
                preprocess_time = time.time() - preprocess_start
                total_preprocessing_time += preprocess_time
                
                # Métricas de detecção
                detect_start = time.time()
                frame_detections = self.detect_objects(processed_image, threshold)
                detect_time = time.time() - detect_start
                total_detection_time += detect_time
                
                # Análise das detecções
                metrics["detection"]["total_detections"] += len(frame_detections)
                for det in frame_detections:
                    # Distribuição de confiança
                    conf = det["score"]
                    for range_key in metrics["detection"]["confidence_distribution"].keys():
                        min_conf, max_conf = map(float, range_key.split("-"))
                        if min_conf <= conf < max_conf:
                            metrics["detection"]["confidence_distribution"][range_key] += 1
                    
                    # Tamanho das detecções
                    box = det["box"]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    area = width * height
                    if area < 32*32:
                        metrics["detection"]["detection_sizes"]["small"] += 1
                    elif area < 96*96:
                        metrics["detection"]["detection_sizes"]["medium"] += 1
                    else:
                        metrics["detection"]["detection_sizes"]["large"] += 1
                    
                    # Adicionar timestamp à detecção
                    det["timestamp"] = timestamp
                    all_detections.append(det)
                
                # Métricas de memória
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
            
            # Calcular médias e estatísticas finais
            total_time = time.time() - start_time
            num_frames = len(frames)
            
            metrics["performance"].update({
                "total_time": total_time,
                "fps_processing": num_frames / total_time,
                "avg_detection_time": total_detection_time / num_frames,
                "avg_preprocessing_time": total_preprocessing_time / num_frames
            })
            
            metrics["preprocessing"]["avg_image_size"].update({
                "width": metrics["preprocessing"]["avg_image_size"]["width"] / num_frames,
                "height": metrics["preprocessing"]["avg_image_size"]["height"] / num_frames
            })
            
            metrics["preprocessing"]["brightness_stats"]["avg"] /= num_frames
            metrics["preprocessing"]["contrast_stats"]["avg"] /= num_frames
            
            metrics["memory"].update({
                "peak_memory_mb": max(memory_samples),
                "avg_memory_mb": sum(memory_samples) / len(memory_samples)
            })
            
            # Adicionar detecções às métricas
            metrics["detections"] = all_detections
            
            # Gerar vídeo com as detecções
            output_path = self.generate_output_video(video_path, all_detections, fps=fps)
            
            return output_path, metrics
            
        except Exception as e:
            print(f"Erro ao processar vídeo: {str(e)}")
            traceback.print_exc()
            return video_path, {}

class WeaponDetector:
    def __init__(self):
        """Initialize using the Singleton instance."""
        self._detector = WeaponDetectorSingleton()
    
    @property
    def device(self):
        return self._detector.device
    
    @property
    def owlv2_processor(self):
        return self._detector.owlv2_processor
    
    @property
    def owlv2_model(self):
        return self._detector.owlv2_model
    
    @property
    def thread_pool(self):
        return self._detector.thread_pool
    
    @property
    def dangerous_objects(self):
        return self._detector.dangerous_objects
    
    @property
    def text_inputs(self):
        return self._detector.text_inputs
    
    def _get_cache_key(self, video_path: str, fps: int, threshold: float) -> str:
        """Generate cache key based on video content and parameters."""
        hasher = hashlib.sha256()
        with open(video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        params = f"{fps}_{threshold}"
        hasher.update(params.encode())
        return hasher.hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self._detector.video_cache_dir / f"{cache_key}.cache"
    
    @torch.inference_mode()
    def detect_objects(self, image: Image.Image, threshold: float = 0.3) -> List[Dict]:
        """Detect objects in image using pre-processed queries."""
        return self._detector.detect_objects(image, threshold)
    
    def extract_frames(self, video_path: str, fps: int = 2) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video using ffmpeg."""
        return self._detector.extract_frames(video_path, fps)
    
    def analyze_video(self, video_path: str, threshold: float = 0.3, fps: int = 5, cancel_event=None) -> Tuple[List[Dict], Dict, Dict]:
        """Analyze video for dangerous objects."""
        return self._detector.analyze_video(video_path, threshold, fps, cancel_event)
    
    def generate_output_video(self, video_path: str, detections: List[Dict], output_path: str = None, fps: int = 30) -> str:
        """Gera vídeo final usando apenas os frames analisados."""
        return self._detector.generate_output_video(video_path, detections, output_path, fps)
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Desenha os bounding boxes nos frames."""
        return self._detector.draw_bounding_boxes(frame, detections)
    
    def generate_ffmpeg_filter(self, detections, fps):
        """
        Gera um filtro ffmpeg para desenhar bounding boxes no vídeo.
        """
        return self._detector.generate_ffmpeg_filter(detections, fps)
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load results from cache if available and valid."""
        return self._detector._load_from_cache(cache_key)
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save results to cache with metadata."""
        return self._detector._save_to_cache(cache_key, data)
    
    def _create_time_ranges(self, timestamps: List[float]) -> List[Dict]:
        """Create time ranges with optimized gap threshold."""
        return self._detector._create_time_ranges(timestamps)
    
    def process_video(self, video_path: str, fps: int = None, threshold: float = 0.3) -> Tuple[str, Dict]:
        """Process video and return path to analyzed video and technical details."""
        return self._detector.process_video(video_path, fps, threshold)
