import os
from dotenv import load_dotenv
from src.presentation.web.gradio_interface import GradioInterface
import logging
import torch

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Função principal que inicia a aplicação."""
    try:
        # Verificar se está rodando no Hugging Face
        IS_HUGGINGFACE = os.getenv('SPACE_ID') is not None
        
        # Carregar configurações do ambiente apropriado
        if IS_HUGGINGFACE:
            load_dotenv('.env.huggingface')
            logger.info("Ambiente HuggingFace detectado")
        else:
            load_dotenv('.env')
            logger.info("Ambiente local detectado")
        
        # Criar e configurar interface
        interface = GradioInterface()
        demo = interface.create_interface()
        
        if IS_HUGGINGFACE:
            # Calcular número ideal de workers baseado na GPU
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # em GB
                max_concurrent = min(2, int(gpu_mem / 8))  # 8GB por worker
                logger.info(f"GPU Memory: {gpu_mem:.1f}GB, Max Concurrent: {max_concurrent}")
            else:
                max_concurrent = 1
            
            # Primeiro configurar a fila
            demo = demo.queue(
                max_size=16,  # Aumentado para corresponder ao max_batch_size
                concurrency_limit=max_concurrent,  # Baseado na memória GPU
                status_update_rate=10,  # Atualizações mais frequentes
                api_open=False,
                max_batch_size=16  # Aumentado para corresponder ao detector
            )
            # Depois fazer o launch
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                max_threads=4  # Limitar threads da CPU
            )
        else:
            # Ambiente local - apenas launch direto
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True
            )
            
    except Exception as e:
        logger.error(f"Erro ao iniciar aplicação: {str(e)}")
        raise

if __name__ == "__main__":
    main() 