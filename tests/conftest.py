import pytest
import os
import sys
from pathlib import Path

# Adiciona o diretório src ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_video_path():
    """Retorna o caminho para um vídeo de teste"""
    return str(Path(__file__).parent / "fixtures" / "sample_video.mp4")

@pytest.fixture
def mock_weapon_detector_service():
    """Mock do serviço de detecção de armas"""
    class MockWeaponDetectorService:
        def detect(self, video_path, threshold=0.5):
            return {
                "detections": [
                    {"label": "weapon", "confidence": 0.8, "bbox": [10, 10, 100, 100]},
                ],
                "frame_count": 30,
                "processing_time": 1.5
            }
    
    return MockWeaponDetectorService()

@pytest.fixture
def mock_notification_service():
    """Mock do serviço de notificação"""
    class MockNotificationService:
        def send_notification(self, message, level="info"):
            return {"status": "success", "message": message}
    
    return MockNotificationService()

@pytest.fixture
def mock_system_monitor():
    """Mock do monitor de sistema"""
    class MockSystemMonitor:
        def get_system_info(self):
            return {
                "cpu_percent": 50.0,
                "memory_percent": 60.0,
                "gpu_info": {"name": "Test GPU", "memory_used": 1000}
            }
    
    return MockSystemMonitor() 