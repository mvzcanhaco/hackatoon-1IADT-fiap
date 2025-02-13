import gradio as gr
import os
from typing import Tuple, Any
from pathlib import Path
from src.application.use_cases.process_video import ProcessVideoUseCase, ProcessVideoRequest
from src.infrastructure.services.weapon_detector import WeaponDetectorService
from src.infrastructure.services.notification_services import NotificationServiceFactory
import logging
from huggingface_hub import hf_hub_download, HfApi
import tempfile

logger = logging.getLogger(__name__)

class GradioInterface:
    """Interface Gradio usando Clean Architecture."""
    
    def __init__(self):
        self.detector = WeaponDetectorService()
        self.notification_factory = NotificationServiceFactory()
        self.default_fps = 2 if self.detector.device_type == "GPU" else 1
        self.default_resolution = "640" if self.detector.device_type == "GPU" else "480"
        self.is_huggingface = os.getenv('SPACE_ID') is not None
        
        # Configurar dataset apenas no ambiente Hugging Face
        if self.is_huggingface:
            self.dataset_id = "marcuscanhaco/weapon-test"
            self.cache_dir = os.path.join(tempfile.gettempdir(), 'weapon_detection_videos')
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Configurar API do Hugging Face
            self.hf_token = os.getenv('HF_TOKEN')
            self.api = HfApi(token=self.hf_token)
            
            # Listar arquivos do dataset
            try:
                files = self.api.list_repo_files(self.dataset_id, repo_type="dataset")
                self.sample_videos = [
                    {
                        'path': f,
                        'name': Path(f).stem.replace('_', ' ').title(),
                        'ground_truth': '🚨 Vídeo de Teste'
                    }
                    for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
                ]
                logger.info(f"Encontrados {len(self.sample_videos)} vídeos no dataset")
            except Exception as e:
                logger.error(f"Erro ao listar arquivos do dataset: {str(e)}")
                self.sample_videos = []
        
        self.use_case = ProcessVideoUseCase(
            detector=self.detector,
            notification_factory=self.notification_factory,
            default_fps=self.default_fps,
            default_resolution=int(self.default_resolution)
        )
    
    def _download_video(self, video_path: str) -> str:
        """Baixa um vídeo do dataset e retorna o caminho local."""
        try:
            local_path = hf_hub_download(
                repo_id=self.dataset_id,
                filename=video_path,
                repo_type="dataset",
                local_dir=self.cache_dir,
                token=self.hf_token,
                local_dir_use_symlinks=False
            )
            logger.info(f"Vídeo baixado com sucesso: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Erro ao baixar vídeo {video_path}: {str(e)}")
            return ""
    
    def list_sample_videos(self) -> list:
        """Lista os vídeos de exemplo do dataset ou da pasta local."""
        try:
            if self.is_huggingface:
                logger.info("Ambiente Hugging Face detectado")
                videos = []
                for video in self.sample_videos:
                    local_path = self._download_video(video['path'])
                    if local_path:
                        videos.append({
                            'path': local_path,
                            'name': video['name'],
                            'ground_truth': video['ground_truth']
                        })
                return videos
            else:
                logger.info("Ambiente local detectado, usando pasta videos")
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                videos = []
                base_dir = Path("videos")
                if not base_dir.exists():
                    os.makedirs(base_dir)
                    logger.info(f"Diretório videos criado: {base_dir}")
                
                for ext in video_extensions:
                    for video_path in base_dir.glob(f'*{ext}'):  # Removido o glob recursivo
                        videos.append({
                            'path': str(video_path),
                            'name': video_path.name,
                            'ground_truth': '📼 Vídeo de Teste'
                        })
                
                return videos
            
        except Exception as e:
            logger.error(f"Erro ao listar vídeos: {str(e)}")
            return []

    def load_sample_video(self, video_path: str) -> str:
        """Carrega um vídeo de exemplo."""
        try:
            if not video_path:
                return ""
            
            if os.path.exists(video_path):
                logger.info(f"Carregando vídeo: {video_path}")
                return video_path
                
            logger.warning(f"Vídeo não encontrado: {video_path}")
            return ""
        except Exception as e:
            logger.error(f"Erro ao carregar vídeo: {str(e)}")
            return ""
    
    def create_interface(self) -> gr.Blocks:
        """Cria a interface Gradio."""
        title = "Detector de Riscos em Vídeos"
        sample_videos = self.list_sample_videos()
        
        with gr.Blocks(
            title=title,
            theme=gr.themes.Ocean(),
            css="footer {display: none !important}"
        ) as demo:
            gr.Markdown(f"""# 🚨 {title}
            
            Faça upload de um vídeo para detectar objetos perigosos.
            Opcionalmente, configure notificações para receber alertas em caso de detecções.

            **Importante para melhor performance:**
            - Vídeos de até 60 segundos
            - FPS entre 1-2 para análise com maior performance
            - FPS maior que 2 para análise com maior precisão
            """)
            with gr.Group():
                gr.Markdown("""### Configuração de Processamento""")
                with gr.Row():
                    threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="Limiar de Detecção",
                    )
                    fps = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=self.default_fps,
                        step=1,
                        label="Frames por Segundo",
                    )
                    resolution = gr.Radio(
                        choices=["480", "640", "768"],
                        value=self.default_resolution,
                        label="Resolução de Processamento",
                )
            with gr.Group():
                gr.Markdown("""### Configuração de Notificações de Detecção (Opcional)""")
                with gr.Row():
                    notification_type = gr.Radio(
                        choices=self.notification_factory.get_available_services(),
                        value="email",
                        label="Tipo de Notificação",
                        interactive=True,
                    )
                    notification_target = gr.Textbox(
                        label="Destino da Notificação (E-mail)",
                        placeholder="exemplo@email.com",
                    )
            with gr.Row():
                with gr.Column(scale=2):
                    input_video = gr.Video(
                        label="Vídeo de Entrada",
                        format="mp4",
                        interactive=True,
                        height=400
                    )
                
                    submit_btn = gr.Button(
                        "Analisar Vídeo",
                        variant="primary",
                        scale=2
                    )
                with gr.Column(scale=1):
                    status = gr.Textbox(
                        label="Status da Detecção",
                        lines=4,
                        show_copy_button=True
                    )
                    with gr.Accordion("Detalhes Técnicos", open=False):
                        json_output = gr.JSON(
                            label="Detalhes Técnicos",
                        )
                    
                    # Informações adicionais
                    with gr.Accordion("Informações Adicionais", open=False):
                        gr.Markdown("""
                        ### Sobre o Detector
                        Este sistema utiliza um modelo de IA avançado para detectar objetos perigosos em vídeos.
                        
                        ### Tipos de Objetos Detectados
                        - Armas de fogo (pistolas, rifles, etc.)
                        - Armas brancas (facas, canivetes, etc.)
                        - Objetos perigosos (bastões, objetos pontiagudos, etc.)
                        
                        ### Recomendações
                        - Use vídeos com boa iluminação
                        - Evite vídeos muito longos
                        - Mantenha os objetos visíveis e em foco
                        """)
            # Vídeos de exemplo
            if sample_videos:
                gr.Markdown("### Vídeos de Exemplo")
                examples = [
                    [video['path']] for video in sample_videos
                ]
                gr.Examples(
                    examples=examples,
                    inputs=input_video,
                    outputs=input_video,
                    fn=self.load_sample_video,
                    label="Clique em um vídeo para carregá-lo"
                )
            
            # Configurar callback do botão
            submit_btn.click(
                fn=lambda *args: self._process_video(*args),
                inputs=[
                    input_video,
                    threshold,
                    fps,
                    resolution,
                    notification_type,
                    notification_target
                ],
                outputs=[status, json_output]
            )
        
        return demo
    
    def _process_video(
        self,
        video_path: str,
        threshold: float = 0.5,
        fps: int = None,
        resolution: str = None,
        notification_type: str = None,
        notification_target: str = None
    ) -> Tuple[str, Any]:
        """Processa o vídeo usando o caso de uso."""
        if not video_path:
            return "Erro: Nenhum vídeo fornecido", {}
            
        # Usar valores padrão se não especificados
        fps = fps or self.default_fps
        resolution = resolution or self.default_resolution
        
        request = ProcessVideoRequest(
            video_path=video_path,
            threshold=threshold,
            fps=fps,
            resolution=int(resolution),
            notification_type=notification_type,
            notification_target=notification_target
        )
        
        response = self.use_case.execute(request)
        
        return (
            response.status_message,
            response.detection_result.__dict__
        ) 