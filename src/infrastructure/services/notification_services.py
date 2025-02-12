import logging
import os
from typing import Dict, Any
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from src.domain.interfaces.notification import NotificationService, NotificationFactory

logger = logging.getLogger(__name__)

class EmailNotification(NotificationService):
    def send_notification(self, detection_data: Dict[str, Any], recipient: str) -> bool:
        try:
            # Verificar se há detecções
            if not detection_data.get("detections"):
                logger.info("Nenhuma detecção para notificar")
                return True  # Retorna True pois não é um erro
                
            sender_email = os.getenv('NOTIFICATION_EMAIL')
            sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
            
            if not sender_email:
                logger.error("NOTIFICATION_EMAIL não configurado")
                return False
                
            if not sendgrid_api_key:
                logger.error("SENDGRID_API_KEY não configurada")
                return False
                
            if not recipient:
                logger.error("Destinatário de e-mail não fornecido")
                return False
                
            body = self._format_email_body(detection_data)
            
            message = Mail(
                from_email=sender_email,
                to_emails=recipient,
                subject='🚨 ALERTA DE SEGURANÇA - Detecção de Risco',
                html_content=f'<pre style="font-family: monospace;">{body}</pre>'
            )
            
            try:
                sg = SendGridAPIClient(sendgrid_api_key)
                response = sg.send(message)
                success = response.status_code == 202
                
                if success:
                    logger.info(f"E-mail enviado com sucesso para {recipient}")
                    logger.debug(f"Status: {response.status_code}")
                    logger.debug(f"Body: {response.body}")
                    logger.debug(f"Headers: {response.headers}")
                else:
                    logger.error(f"Erro ao enviar e-mail. Status code: {response.status_code}")
                
                return success
                
            except Exception as e:
                logger.error(f"Erro ao enviar e-mail via SendGrid: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Erro no serviço de e-mail: {str(e)}")
            return False
            
    def _format_email_body(self, detection_data: Dict[str, Any]) -> str:
        """Formata o corpo do e-mail com os dados da detecção."""
        try:
            detections = detection_data.get("detections", [])
            if not detections:
                return "Nenhuma detecção encontrada no vídeo."
                
            body = """
⚠️ ALERTA DE SEGURANÇA ⚠️

Uma detecção de risco foi identificada:

"""
            # Adicionar informações da primeira detecção
            first_detection = detections[0]
            body += f"""📹 Detecção:
- Objeto: {first_detection.get('label', 'Desconhecido')}
- Confiança: {first_detection.get('confidence', 0):.2%}
- Timestamp: {first_detection.get('timestamp', 0):.2f}s

"""
            
            # Adicionar informações técnicas
            if "technical" in detection_data:
                tech = detection_data["technical"]
                body += f"""Informações Técnicas:
- Threshold: {tech.get('threshold', 'N/A')}
- FPS: {tech.get('fps', 'N/A')}
- Resolução: {tech.get('resolution', 'N/A')}
"""
            
            body += """
--
Este é um e-mail automático enviado pelo Sistema de Detecção de Riscos.
Não responda este e-mail.
"""
            
            return body
            
        except Exception as e:
            logger.error(f"Erro ao formatar e-mail: {str(e)}")
            return "Erro ao formatar dados da detecção."

class NotificationServiceFactory(NotificationFactory):
    def __init__(self):
        self._services = {'email': EmailNotification()}
    
    def create_service(self, service_type: str) -> NotificationService:
        return self._services.get(service_type)
    
    def get_available_services(self) -> list:
        return list(self._services.keys())