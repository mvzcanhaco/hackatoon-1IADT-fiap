import gradio as gr
import tempfile
import os
from weapon_detection import WeaponDetector
import json
from pathlib import Path

# Inicializar detector
print("Initializing weapon detector...")
weapon_detector = WeaponDetector()

def process_video(video_path, threshold=0.5, fps=2):
    """Process video and return detections with analyzed video."""
    if video_path is None:
        return None, "No video uploaded", None
    
    try:
        # Se o vídeo for um arquivo temporário do Gradio, ele será uma string
        if isinstance(video_path, str):
            video_file = video_path
        else:
            video_file = video_path.name
        
        # Processar vídeo
        detections, summary, metrics = weapon_detector.analyze_video(
            video_file,
            threshold=threshold,
            fps=fps
        )
        
        # Formatar resultados de forma simples
        if detections:
            result_text = "RISCO DETECTADO!\n\n"
            if summary.get("time_ranges"):
                result_text += "Intervalos de Risco:\n"
                for range_info in summary["time_ranges"]:
                    result_text += f"• {range_info['start']:.1f}s até {range_info['end']:.1f}s\n"
        else:
            result_text = "✅ Nenhum risco detectado"
        
        # Calcular FPS efetivo com proteção contra divisão por zero
        video_duration = metrics.get("video_duration", 0)
        frames_analyzed = metrics.get("frames_analyzed", 0)
        effective_fps = "N/A"
        if video_duration > 0:
            effective_fps = f"{frames_analyzed/video_duration:.1f}"
        
        # Expandir detalhes técnicos
        technical_details = {
            "status": summary["risk_status"],
            "device_info": {
                "type": metrics["device_type"],
                "optimization": "CUDA enabled" if metrics["device_type"] == "cuda" else "CPU/MPS optimized"
            },
            "performance_metrics": {
                "total_time": f"{metrics['total_time']:.2f}s",
                "frame_extraction_time": f"{metrics['frame_extraction_time']:.2f}s",
                "analysis_time": f"{metrics['analysis_time']:.2f}s",
                "frames_analyzed": frames_analyzed,
                "fps_target": fps,
                "detection_threshold": threshold
            },
            "video_info": {
                "duration": f"{video_duration:.2f}s",
                "total_frames": frames_analyzed,
                "effective_fps": effective_fps
            },
            "detection_details": {
                "total_detections": len(detections),
                "unique_timestamps": len(set(d["timestamp"] for d in detections)),
                "detection_types": list(set(d["type"] for d in detections)),
                "time_ranges": summary["time_ranges"]
            }
        }
        
        # Retornar vídeo analisado se disponível
        output_video = summary.get("output_video")
        if output_video and os.path.exists(output_video):
            print(f"Returning analyzed video: {output_video}")
            return output_video, result_text, json.dumps(technical_details, indent=2)
        else:
            print("No analyzed video available, returning original")
            return video_file, result_text, json.dumps(technical_details, indent=2)
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return video_path, f"Error: {str(e)}", None

# Interface Gradio
with gr.Blocks(title="Weapon Detection System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚨 Sistema de Detecção de Riscos
    Faça upload de um vídeo para detectar objetos perigosos.
    """)
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Vídeo de Entrada")
            with gr.Row():
                threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Limiar de Detecção")
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

# Iniciar interface
if __name__ == "__main__":
    demo.launch(share=False)