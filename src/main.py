import os
from dotenv import load_dotenv
from src.presentation.web.gradio_interface import GradioInterface
import logging
import torch
import gc
import nvidia_smi
from src.domain.factories.detector_factory import force_gpu_init, is_gpu_available

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_type():
    """Verifica o tipo de GPU disponível no ambiente Hugging Face."""
    try:
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvidia_smi.nvmlDeviceGetName(handle)
        total_memory = info.total / (1024**3)  # Converter para GB
        
        logger.info(f"GPU detectada: {gpu_name}")
        logger.info(f"Memória total: {total_memory:.2f}GB")
        
        # T4 dedicada tem tipicamente 16GB
        if "T4" in gpu_name and total_memory > 14:
            return "t4_dedicated"
        # Zero-GPU compartilhada tem tipicamente menos memória
        elif total_memory < 14:
            return "zero_gpu_shared"
        else:
            return "unknown"
            
    except Exception as e:
        logger.error(f"Erro ao verificar tipo de GPU: {str(e)}")
        return "unknown"
    finally:
        try:
            nvidia_smi.nvmlShutdown()
        except:
            pass

def setup_gpu_environment(gpu_type: str) -> bool:
    """Configura o ambiente GPU com base no tipo detectado."""
    try:
        # Verificar ambiente CUDA
        if not torch.cuda.is_available():
            logger.warning("CUDA não está disponível")
            return False
            
        # Configurações comuns
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Limpar memória
        torch.cuda.empty_cache()
        gc.collect()
        
        if gpu_type == "t4_dedicated":
            # Configurações otimizadas para T4 dedicada
            logger.info("Configurando para T4 dedicada")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            # Aumentar fração de memória e tamanho do split
            torch.cuda.set_per_process_memory_fraction(0.95)  # Aumentado para 95%
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'  # Aumentado para 1GB
            
        elif gpu_type == "zero_gpu_shared":
            # Configurações para Zero-GPU compartilhada
            logger.info("Configurando para Zero-GPU compartilhada")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # Limitar uso de memória
            torch.cuda.set_per_process_memory_fraction(0.6)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
        # Verificar configuração
        try:
            device = torch.device('cuda')
            dummy = torch.zeros(1, device=device)
            del dummy
            logger.info(f"Configurações GPU aplicadas com sucesso para: {gpu_type}")
            return True
        except Exception as e:
            logger.error(f"Erro ao configurar GPU: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Erro ao configurar ambiente GPU: {str(e)}")
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
            
            # Identificar e configurar GPU
            gpu_type = check_gpu_type()
            gpu_available = setup_gpu_environment(gpu_type)
            
            if gpu_available:
                logger.info(f"GPU configurada com sucesso: {gpu_type}")
            else:
                logger.warning("GPU não disponível ou não configurada corretamente")
        else:
            load_dotenv('.env')
            logger.info("Ambiente local detectado")
            gpu_available = False
        
        # Criar e configurar interface
        interface = GradioInterface()
        demo = interface.create_interface()
        
        if IS_HUGGINGFACE:
            # Configurar com base no tipo de GPU
            if gpu_type == "t4_dedicated":
                max_concurrent = 2  # T4 pode lidar com mais requisições
                queue_size = 10
            else:
                max_concurrent = 1  # Zero-GPU precisa ser mais conservadora
                queue_size = 5
            
            # Configurar fila
            demo = demo.queue(
                api_open=False,
                max_size=queue_size,
                status_update_rate="auto",
                concurrency_count=max_concurrent
            )
            
            # Launch
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                max_threads=max_concurrent
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