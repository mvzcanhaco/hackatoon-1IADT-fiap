import gradio as gr
import tempfile
import os
from weapon_detection import WeaponDetector
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import shutil
import traceback
import cv2
import torch
import time

# Inicializar detector
print("Initializing weapon detector...")
weapon_detector = WeaponDetector()

def format_detections(detections, fps):
    """Formata as detecções para JSON incluindo confiança."""
    if not detections:
        return {"message": "Nenhuma detecção encontrada", "detections": []}
    
    # Agrupar detecções por timestamp
    detections_by_time = {}
    for det in detections:
        t = det.get("timestamp", 0)
        if t not in detections_by_time:
            detections_by_time[t] = []
        detections_by_time[t].append(det)
    
    # Criar ranges de tempo
    detector = WeaponDetector()
    time_ranges = detector._create_time_ranges(list(detections_by_time.keys()))
    
    formatted_output = {
        "message": "Detecções encontradas nos seguintes intervalos:",
        "detections": []
    }
    
    for time_range in time_ranges:
        start_time = time_range["start"]
        end_time = time_range["end"]
        
        # Coletar todas as detecções neste intervalo
        range_detections = []
        for t in detections_by_time.keys():
            if start_time <= t <= end_time:
                for det in detections_by_time[t]:
                    range_detections.append({
                        "timestamp": t,
                        "box": det["box"],
                        "confidence": round(det["score"] * 100, 2),  # Confiança em porcentagem
                        "type": det.get("label", "RISCO")
                    })
        
        # Ordenar por confiança
        range_detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        formatted_output["detections"].append({
            "time_range": {
                "start": f"{int(start_time // 60):02d}:{int(start_time % 60):02d}",
                "end": f"{int(end_time // 60):02d}:{int(end_time % 60):02d}",
                "start_seconds": round(start_time, 2),
                "end_seconds": round(end_time, 2)
            },
            "objects": range_detections
        })
    
    return formatted_output

def process_video(video_path, threshold=0.4, fps=2):
    """Process video and return detections with analyzed video."""
    if video_path is None:
        return None, "Erro: Nenhum vídeo fornecido", None
        
    try:
        start_time = time.time()
        
        # Processar vídeo
        detector = WeaponDetector()
        output_path, metrics = detector.process_video(video_path, fps=fps, threshold=threshold)
        
        # Extrair informações do vídeo
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calcular métricas
        total_time = time.time() - start_time
        frame_extraction_time = metrics.get("performance", {}).get("avg_preprocessing_time", 0) * metrics.get("performance", {}).get("frames_processed", 0)
        analysis_time = total_time - frame_extraction_time
        
        # Organizar detecções por timestamp
        detections = metrics.get("detections", [])
        detections_by_time = {}
        detection_types = set()
        
        for det in detections:
            t = det.get("timestamp", 0)
            if t not in detections_by_time:
                detections_by_time[t] = []
            det_info = {
                "box": det["box"],
                "score": round(det["score"] * 100, 1),  # Converter para porcentagem
                "type": det.get("label", "RISCO")
            }
            detections_by_time[t].append(det_info)
            detection_types.add(det.get("label", "RISCO"))
        
        # Criar ranges de tempo
        time_ranges = detector._create_time_ranges(list(detections_by_time.keys()))
        
        # Formatar saída JSON
        json_output = {
            "status": "danger" if detections else "safe",
            "device_info": {
                "type": "mps" if torch.backends.mps.is_available() else "cpu",
                "optimization": "CPU/MPS optimized"
            },
            "performance_metrics": {
                "total_time": f"{total_time:.2f}s",
                "frame_extraction_time": f"{frame_extraction_time:.2f}s",
                "analysis_time": f"{analysis_time:.2f}s",
                "frames_analyzed": metrics.get("performance", {}).get("frames_processed", 0),
                "fps_target": fps,
                "detection_threshold": threshold
            },
            "video_info": {
                "duration": f"{duration:.2f}s",
                "total_frames": total_frames,
                "effective_fps": f"{total_frames/duration:.1f}"
            },
            "detection_details": {
                "total_detections": len(detections),
                "unique_timestamps": len(detections_by_time),
                "detection_types": list(detection_types),
                "time_ranges": time_ranges,
                "detections_by_time": {
                    str(t): dets for t, dets in detections_by_time.items()
                }
            }
        }
        
        # Gerar mensagem de status
        status_msg = f"""Processamento concluído em {total_time:.2f}s!
        
        Detecções: {len(detections)}
        Frames analisados: {metrics.get('performance', {}).get('frames_processed', 0)}
        FPS efetivo: {total_frames/duration:.1f}
        
        Status: {'⚠️ RISCO DETECTADO' if detections else '✅ SEGURO'}
        """
        
        return output_path, status_msg, json_output
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
        return video_path, f"Error: {str(e)}", None

# Interface Gradio
app = Flask(__name__)
with gr.Blocks(title="Weapon Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚨 Sistema de Detecção de Riscos
    Faça upload de um vídeo para detectar objetos perigosos.
    """)
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Vídeo de Entrada")
            with gr.Row():
                threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.4, step=0.1, label="Limiar de Detecção")
                fps = gr.Slider(minimum=1, maximum=30, value=2, step=1, label="Frames por Segundo")
            submit_btn = gr.Button("Analisar Vídeo", variant="primary")
        
        with gr.Column():
            output_video = gr.Video(label="Vídeo Analisado (com detecções)")
            output_text = gr.Textbox(label="Resultado da Análise", lines=5)
            json_output = gr.JSON(label="Detalhes Técnicos")
    
    submit_btn.click(
        process_video,
        inputs=[input_video, threshold, fps],
        outputs=[output_video, output_text, json_output]
    )

@app.route('/process_video', methods=['POST'])
def process_video_api():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'No selected video file'}), 400
        
        # Salvar vídeo temporariamente
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, secure_filename(video.filename))
        video.save(video_path)
        
        # Processar vídeo
        detector = WeaponDetector()
        output_path, metrics = detector.process_video(video_path)
        
        # Formatar saída JSON com detecções e métricas
        json_output = {
            "detections": format_detections(metrics.get("detections", []), metrics.get("fps", 2)),
            "technical_metrics": {
                "performance": metrics.get("performance", {}),
                "detection_stats": metrics.get("detection", {}),
                "preprocessing": metrics.get("preprocessing", {}),
                "memory": metrics.get("memory", {}),
                "technical": metrics.get("technical", {})
            }
        }
        
        # Mensagem de status
        status_msg = f"""Processamento concluído!
        
        Métricas:
        - Frames processados: {metrics.get('performance', {}).get('frames_processed', 0)}
        - FPS médio: {metrics.get('performance', {}).get('fps_processing', 0):.2f}
        - Detecções totais: {metrics.get('detection', {}).get('total_detections', 0)}
        - Tempo total: {metrics.get('performance', {}).get('total_time', 0):.2f}s
        
        Detalhes completos disponíveis no JSON output."""
        
        return jsonify({
            "output_path": output_path,
            "status_msg": status_msg,
            "json_output": json_output
        })
        
    except Exception as e:
        print(f"Erro ao processar vídeo: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Limpar arquivos temporários
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Erro ao limpar arquivos temporários: {str(e)}")

@app.route('/metrics', methods=['GET'])
def get_metrics_api():
    """Retorna as métricas técnicas do último processamento."""
    detector = WeaponDetector()
    return jsonify(detector.get_last_metrics())

# Iniciar interface
if __name__ == "__main__":
    demo.launch(share=False)