from abc import ABC, abstractmethod
from typing import Dict, Any, List

class NotificationService(ABC):
    """Interface base para serviços de notificação."""
    
    @abstractmethod
    def send_notification(self, detection_data: Dict[str, Any], recipient: str) -> bool:
        """Envia notificação usando o serviço específico."""
        pass

class NotificationFactory(ABC):
    """Interface para fábrica de serviços de notificação."""
    
    @abstractmethod
    def create_service(self, service_type: str) -> NotificationService:
        """Cria uma instância do serviço de notificação especificado."""
        pass
    
    @abstractmethod
    def get_available_services(self) -> List[str]:
        """Retorna lista de serviços de notificação disponíveis."""
        pass 