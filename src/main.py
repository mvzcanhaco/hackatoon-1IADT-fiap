import os
from dotenv import load_dotenv
from src.presentation.web.gradio_interface import GradioInterface
import logging
import torch
import gc
from src.domain.factories.detector_factory import force_gpu_init, is_gpu_available

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_zero_gpu():
    """Configurações otimizadas para Zero-GPU."""
    try:
        # Forçar inicialização da GPU
        if is_gpu_available():
            force_gpu_init()
            # Limpar cache CUDA
            torch.cuda.empty_cache()
            gc.collect()
        
            # Configurações para otimizar memória
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Configurar device map para melhor utilização da memória
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            torch.cuda.set_per_process_memory_fraction(0.9)  # Usar 90% da memória disponível
            
            logger.info(f"Configurações Zero-GPU aplicadas com sucesso na GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU não disponível para configuração Zero-GPU. O sistema operará em modo CPU.")
    except Exception as e:
        logger.error(f"Erro ao configurar Zero-GPU: {str(e)}")
        logger.warning("Fallback para modo CPU devido a erro na configuração da GPU.")

def main():
    """Função principal que inicia a aplicação."""
    try:
        # Verificar se está rodando no Hugging Face
        IS_HUGGINGFACE = os.getenv('SPACE_ID') is not None
        
        # Carregar configurações do ambiente apropriado
        if IS_HUGGINGFACE:
            load_dotenv('.env.huggingface')
            logger.info("Ambiente HuggingFace detectado")
            setup_zero_gpu()
        else:
            load_dotenv('.env')
            logger.info("Ambiente local detectado")
        
        # Criar e configurar interface
        interface = GradioInterface()
        demo = interface.create_interface()
        
        if IS_HUGGINGFACE:
            # Calcular número ideal de workers baseado na GPU
            if is_gpu_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # em GB
                max_concurrent = 1  # Forçar single worker para Zero-GPU
                logger.info(f"GPU Memory: {gpu_mem:.1f}GB, Max Concurrent: {max_concurrent}")
            else:
                max_concurrent = 1
                logger.warning("GPU não disponível. O sistema está operando em modo CPU. " +
                             "Todas as funcionalidades estão disponíveis, mas o processamento será mais lento.")
            
            # Primeiro configurar a fila
            demo = demo.queue(
                api_open=False,
                status_update_rate="auto",
                max_size=5  # Reduzir tamanho da fila para economizar memória
            )
            # Depois fazer o launch
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                max_threads=2  # Reduzir número de threads
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