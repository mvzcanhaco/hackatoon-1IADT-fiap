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

def check_cuda_environment():
    """Verifica e configura o ambiente CUDA."""
    try:
        # Verificar variáveis de ambiente CUDA
        cuda_path = os.getenv('CUDA_HOME') or os.getenv('CUDA_PATH')
        if not cuda_path:
            logger.warning("Variáveis de ambiente CUDA não encontradas")
            return False
            
        # Verificar se CUDA está disponível no PyTorch
        if not torch.cuda.is_available():
            logger.warning("PyTorch não detectou CUDA")
            return False
            
        # Tentar obter informações da GPU
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detectada: {device_name}")
                return True
        except Exception as e:
            logger.warning(f"Erro ao obter informações da GPU: {str(e)}")
            
        return False
    except Exception as e:
        logger.error(f"Erro ao verificar ambiente CUDA: {str(e)}")
        return False

def setup_zero_gpu():
    """Configurações otimizadas para Zero-GPU."""
    try:
        # Verificar ambiente CUDA primeiro
        if not check_cuda_environment():
            logger.warning("Ambiente CUDA não está configurado corretamente")
            return False
            
        # Tentar inicializar GPU
        if is_gpu_available():
            # Configurar ambiente
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Limpar memória
            torch.cuda.empty_cache()
            gc.collect()
            
            # Configurações de memória e performance
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Configurar fração de memória
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Verificar se a configuração foi bem sucedida
            try:
                device = torch.device('cuda')
                dummy = torch.zeros(1, device=device)
                del dummy
                logger.info(f"Configurações Zero-GPU aplicadas com sucesso na GPU: {torch.cuda.get_device_name(0)}")
                return True
            except Exception as e:
                logger.error(f"Erro ao configurar GPU: {str(e)}")
                return False
        else:
            logger.warning("GPU não disponível para configuração Zero-GPU. O sistema operará em modo CPU.")
            return False
    except Exception as e:
        logger.error(f"Erro ao configurar Zero-GPU: {str(e)}")
        logger.warning("Fallback para modo CPU devido a erro na configuração da GPU.")
        return False

def main():
    """Função principal que inicia a aplicação."""
    try:
        # Verificar se está rodando no Hugging Face
        IS_HUGGINGFACE = os.getenv('SPACE_ID') is not None
        
        # Carregar configurações do ambiente apropriado
        if IS_HUGGINGFACE:
            load_dotenv('.env.huggingface')
            logger.info("Ambiente HuggingFace detectado")
            gpu_available = setup_zero_gpu()
        else:
            load_dotenv('.env')
            logger.info("Ambiente local detectado")
            gpu_available = False
        
        # Criar e configurar interface
        interface = GradioInterface()
        demo = interface.create_interface()
        
        if IS_HUGGINGFACE:
            # Configurar com base na disponibilidade da GPU
            if gpu_available:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                max_concurrent = 1  # Forçar single worker para Zero-GPU
                logger.info(f"GPU Memory: {gpu_mem:.1f}GB, Max Concurrent: {max_concurrent}")
            else:
                max_concurrent = 1
                logger.warning("GPU não disponível. O sistema está operando em modo CPU. " +
                             "Todas as funcionalidades estão disponíveis, mas o processamento será mais lento.")
            
            # Configurar fila
            demo = demo.queue(
                api_open=False,
                status_update_rate="auto",
                max_size=5  # Reduzir tamanho da fila para economizar memória
            )
            
            # Launch
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                max_threads=2  # Reduzir número de threads
            )
        else:
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