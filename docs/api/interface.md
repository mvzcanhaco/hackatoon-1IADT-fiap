# Interface e API

## Interface Web

### Componentes Principais

1. **Upload de Vídeo**
   - Formatos suportados: MP4, AVI, MOV
   - Tamanho máximo: 100MB
   - Duração máxima: 5 minutos

2. **Configurações de Detecção**
   - Limiar de confiança (0-100%)
   - Modo de processamento (GPU/CPU)
   - Tamanho do lote de frames

3. **Visualização de Resultados**
   - Vídeo com anotações
   - Timeline de detecções
   - Métricas de processamento

### Exemplos de Uso

1. **Upload Simples**

   ```python
   with gr.Blocks() as demo:
       video_input = gr.Video()
       detect_btn = gr.Button("Detectar")
       output_display = gr.Video()
   ```

2. **Configurações Avançadas**

   ```python
   with gr.Blocks() as demo:
       with gr.Row():
           confidence = gr.Slider(0, 100)
           batch_size = gr.Number(value=4)
   ```

## API REST

### Endpoints

#### 1. Detecção em Vídeo

```http
POST /api/detect
Content-Type: multipart/form-data

{
    "video": binary_data,
    "confidence": 0.5,
    "use_gpu": true
}
```

**Resposta:**

```json
{
    "detections": [
        {
            "frame": 0,
            "timestamp": "00:00:01",
            "objects": [
                {
                    "label": "arma",
                    "confidence": 95.5,
                    "bbox": [0, 0, 100, 100]
                }
            ]
        }
    ],
    "metrics": {
        "frames_processed": 150,
        "processing_time": 2.5,
        "fps": 60
    }
}
```

#### 2. Status do Serviço

```http
GET /api/status

Response:
{
    "status": "online",
    "gpu_available": true,
    "memory_usage": "45%",
    "queue_size": 2
}
```

### Códigos de Erro

- `400`: Parâmetros inválidos
- `413`: Vídeo muito grande
- `500`: Erro interno
- `503`: Serviço indisponível

### Rate Limiting

- 10 requisições/minuto por IP
- 100 requisições/hora por usuário
- Tamanho máximo do vídeo: 100MB

## Webhooks

### Configuração

```http
POST /api/webhooks/configure
{
    "url": "https://seu-servidor.com/callback",
    "events": ["detection", "error"],
    "secret": "seu_secret"
}
```

### Formato do Callback

```json
{
    "event": "detection",
    "timestamp": "2024-03-20T15:30:00Z",
    "data": {
        "video_id": "abc123",
        "detections": []
    },
    "signature": "hash_hmac"
}
```

## Integração com Outros Serviços

### 1. Hugging Face

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="./video.mp4",
    path_in_repo="videos/test.mp4",
    repo_id="seu-espaco"
)
```

### 2. Sistemas de Notificação

```python
def notify(detection):
    requests.post(
        "https://seu-servidor.com/notify",
        json={"detection": detection}
    )
```

## Considerações de Segurança

1. **Autenticação**
   - Token JWT obrigatório
   - Renovação automática

2. **Rate Limiting**
   - Por IP e usuário
   - Cooldown progressivo

3. **Validação de Entrada**
   - Tamanho máximo
   - Formatos permitidos
   - Sanitização
