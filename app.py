import gradio as gr
import cv2
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from weapon_detection import WeaponDetector
from PIL import Image

# Inicializar o detector de armas uma vez
weapon_detector = WeaponDetector()

def process_frame(frame):
    """
    Processa um frame do vídeo e retorna os dados de análise.
    """
    if frame is None:
        return None
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    analysis = {
        "timestamp": time.time(),
        "mean_intensity": float(np.mean(gray_frame)),
        "min_intensity": float(np.min(gray_frame)),
        "max_intensity": float(np.max(gray_frame))
    }
    
    # Detectar armas no frame
    dangerous_objects = [
        "knife", "machete", "sword", "dagger", "bayonet", "blade",
        "combat knife", "hunting knife", "military knife", "tactical knife",
        "kitchen knife", "butcher knife", "pocket knife", "utility knife",
        "razor", "box cutter", "glass shard", "broken glass", "broken bottle",
        "scissors", "sharp metal", "sharp object", "blade weapon",
        "scalpel", "exacto knife", "craft knife", "paper cutter",
        "ice pick", "awl", "needle", "screwdriver", "metal spike",
        "sharp stick", "sharp pole", "pointed metal", "metal rod",
        "saw blade", "circular saw", "chainsaw", "axe", "hatchet",
        "cleaver", "metal file", "chisel", "wire cutter",
        "sharpened object", "improvised blade", "makeshift weapon",
        "concealed blade", "hidden blade", "modified tool"
    ]
    detections = weapon_detector.detect_objects(Image.fromarray(gray_frame), dangerous_objects)
    
    # Adicionar detecções à análise
    analysis["weapon_detections"] = detections
    
    return analysis

def process_video(video_input):
    """
    Processa o vídeo e retorna os resultados da análise.
    Aceita tanto upload quanto gravação da webcam.
    """
    if video_input is None:
        return "Nenhum vídeo fornecido.", None
        
    # Criar arquivo temporário se o input for um caminho
    if isinstance(video_input, str):
        video_path = video_input
    else:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp.write(video_input)
        video_path = temp.name
        
    cap = cv2.VideoCapture(video_path)
    results = []
    detections_summary = []
    detailed_detections = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        analysis = process_frame(frame)
        if analysis:
            results.append(analysis)
            
            # Adicionar detecções ao resumo e JSON detalhado
            for detection in analysis["weapon_detections"]:
                detections_summary.append(f"{detection['label']} com confiança de {detection['score']*100:.1f}%")
                detailed_detections.append({
                    "tipo": detection["label"],
                    "confiança": detection["score"],
                    "descrição": f"Uma {detection['label']} foi detectada na cena",
                    "caixa": detection["box"]
                })
            
    cap.release()
    
    if not results:
        return "Erro ao processar o vídeo.", None
        
    # Calcular estatísticas gerais
    mean_intensities = [r["mean_intensity"] for r in results]
    stats = {
        "média_intensidade_geral": float(np.mean(mean_intensities)),
        "frames_analisados": len(results),
        "duração_segundos": len(results) / 30.0,  # assumindo 30fps
        "análise_por_frame": results
    }
    
    # Formatar saída para melhor visualização
    output_text = f"""
    Estatísticas do Vídeo:
    - Frames analisados: {stats['frames_analisados']}
    - Duração aproximada: {stats['duração_segundos']:.2f} segundos
    - Média de intensidade: {stats['média_intensidade_geral']:.2f}
    - Detecções: {', '.join(detections_summary) if detections_summary else 'Nenhuma detecção'}
    """
    
    detailed_output = {
        "estatísticas": stats,
        "detecções_detalhadas": detailed_detections
    }
    
    return output_text, json.dumps(detailed_output, indent=2)

# Interface Gradio
with gr.Blocks(title="Processador de Vídeo com Análise") as demo:
    gr.Markdown("# Processador de Vídeo com Análise")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Entrada de Vídeo")
            video_input = gr.Video(
                label="Upload ou Gravação",
                sources=["upload", "webcam"],
                format="mp4"
            )
        
        with gr.Column():
            gr.Markdown("### Resultados da Análise")
            text_output = gr.Textbox(
                label="Resumo",
                lines=4,
                interactive=False
            )
            json_output = gr.JSON(
                label="Dados Detalhados"
            )
    
    # Botão de processamento
    process_btn = gr.Button("Processar Vídeo")
    process_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[text_output, json_output]
    )
    
    gr.Markdown("""
    ### Como Usar
    1. Faça upload de um vídeo ou grave usando a webcam
    2. Clique em "Processar Vídeo"
    3. Veja os resultados da análise no lado direito
    """)

if __name__ == "__main__":
    demo.launch() 