import os
import logging
from src.main import main
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
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
            
        # Iniciar aplicação
        main()
    except Exception as e:
        logger.error(f"Erro ao iniciar aplicação: {str(e)}")
        raise 