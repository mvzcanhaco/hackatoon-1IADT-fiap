import gradio as gr
import os
from typing import Tuple, Any
from pathlib import Path
from src.application.use_cases.process_video import ProcessVideoUseCase, ProcessVideoRequest
from src.infrastructure.services.weapon_detector import WeaponDetectorService
from src.infrastructure.services.notification_services import NotificationServiceFactory
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

class GradioInterface:
    """Interface Gradio usando Clean Architecture."""
    
    def __init__(self):
        self.detector = WeaponDetectorService()
        self.notification_factory = NotificationServiceFactory()
        self.default_fps = 2 if self.detector.device_type == "GPU" else 1
        self.default_resolution = "640" if self.detector.device_type == "GPU" else "480"
        self.dataset_id = "marcuscanhaco/weapon-test"  # ID do dataset de teste
        self.videos_cache_dir = Path(tempfile.gettempdir()) / "weapon_detection_videos"
        self.is_huggingface = os.getenv('SPACE_ID') is not None
        
        self.use_case = ProcessVideoUseCase(
            detector=self.detector,
            notification_factory=self.notification_factory,
            default_fps=self.default_fps,
            default_resolution=int(self.default_resolution)
        )
        
        # Criar diretório de cache se não existir
        if self.is_huggingface:
            self.videos_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Diretório de cache criado em: {self.videos_cache_dir}")
    
    def _download_dataset_videos(self) -> bool:
        """Baixa os vídeos do dataset do Hugging Face."""
        try:
            logger.info(f"Iniciando download do dataset {self.dataset_id}")
            
            # Listar arquivos disponíveis no dataset
            files = list_repo_files(self.dataset_id, repo_type="dataset")
            logger.info(f"Arquivos encontrados no dataset: {files}")
            
            # Baixar cada arquivo de vídeo individualmente
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    try:
                        local_file = hf_hub_download(
                            repo_id=self.dataset_id,
                            filename=file,
                            repo_type="dataset",
                            local_dir=str(self.videos_cache_dir)
                        )
                        logger.info(f"Arquivo baixado com sucesso: {local_file}")
                    except Exception as e:
                        logger.error(f"Erro ao baixar arquivo {file}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao baixar dataset: {str(e)}")
            return False
    
    def list_sample_videos(self) -> list:
        """Lista os vídeos de exemplo do dataset ou da pasta local."""
        try:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            videos = []
            
            if self.is_huggingface:
                logger.info("Ambiente Hugging Face detectado, usando dataset")
                if self._download_dataset_videos():
                    logger.info(f"Procurando vídeos em: {self.videos_cache_dir}")
                    # Listar todos os vídeos no diretório de cache
                    for ext in video_extensions:
                        for video_path in Path(self.videos_cache_dir).rglob(f'*{ext}'):
                            logger.info(f"Vídeo encontrado: {video_path}")
                            videos.append({
                                'path': str(video_path),
                                'name': video_path.name,
                                'ground_truth': '📼 Vídeo de Teste'
                            })
                else:
                    logger.error("Falha ao baixar dataset")
            else:
                logger.info("Ambiente local detectado, usando pasta videos")
                base_dir = Path("videos")
                if not base_dir.exists():
                    os.makedirs(base_dir)
                    logger.info(f"Diretório videos criado: {base_dir}")
                
                # Listar vídeos locais
                for ext in video_extensions:
                    for video_path in base_dir.glob(f'**/*{ext}'):
                        logger.info(f"Vídeo local encontrado: {video_path}")
                        videos.append({
                            'path': str(video_path),
                            'name': video_path.name,
                            'ground_truth': '📼 Vídeo de Teste'
                        })
            
            logger.info(f"Total de vídeos encontrados: {len(videos)}")
            return videos
            
        except Exception as e:
            logger.error(f"Erro ao listar vídeos: {str(e)}")
            return []

    def load_sample_video(self, video_path: str) -> str:
        """Carrega um vídeo de exemplo do cache ou pasta local."""
        try:
            if video_path and os.path.exists(video_path):
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
                with gr.Group():
                    gr.Markdown("### Vídeos de Exemplo")
                    with gr.Row():
                        with gr.Column(scale=4):
                            gr.Markdown("#### Vídeo")
                        with gr.Column(scale=1):
                            gr.Markdown("#### Ação")
                    
                    for video in sample_videos:
                        with gr.Row():
                            with gr.Column(scale=4):
                                gr.Video(
                                    value=video['path'],
                                    format="mp4",
                                    height=150,
                                    interactive=False,
                                    show_label=False
                                )
                            with gr.Column(scale=1, min_width=100):
                                gr.Button(
                                    "📥 Carregar",
                                    size="sm"
                                ).click(
                                    fn=self.load_sample_video,
                                    inputs=[gr.State(video['path'])],
                                    outputs=[input_video]
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