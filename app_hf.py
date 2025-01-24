import gradio as gr
from weapon_detection import WeaponDetector
import os
import json

# Inicializar detector
print("Initializing weapon detector...")
weapon_detector = WeaponDetector()

def process_video(video_path, threshold=0.4, fps=2):
    """Process video and return detections with analyzed video."""
    if video_path is None:
        return None, "Erro: Nenhum vídeo fornecido", None
        
    try:
        # Processar vídeo
        output_path, metrics = weapon_detector.process_video(video_path, fps=fps, threshold=threshold)
        
        # Gerar mensagem de status
        status_msg = f"""Processamento concluído!
        
        Detecções: {len(metrics.get('detections', []))}
        Frames analisados: {metrics.get('performance', {}).get('frames_processed', 0)}
        
        Status: {'⚠️ RISCO DETECTADO' if metrics.get('detections', []) else '✅ SEGURO'}
        """
        
        return output_path, status_msg, json.dumps(metrics, indent=2)
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return video_path, f"Error: {str(e)}", None

# Interface Gradio
with gr.Blocks(title="Detector de Riscos em Vídeos") as demo:
    gr.Markdown("""# 🚨 Detector de Riscos em Vídeos
    
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
            output_video = gr.Video(label="Vídeo Analisado")
            status = gr.Textbox(label="Status", lines=4)
            json_output = gr.JSON(label="Detalhes Técnicos")
    
    submit_btn.click(
        process_video,
        inputs=[input_video, threshold, fps],
        outputs=[output_video, status, json_output]
    )

# Configuração para Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()
