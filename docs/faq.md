# FAQ

## Geral

### Como o sistema funciona?

O sistema utiliza um modelo de IA (OWL-ViT) para detectar objetos de risco em vídeos.
O processamento pode ser feito em GPU ou CPU, com otimizações específicas para cada caso.

### Quais objetos são detectados?

#### Armas de Fogo

- Pistolas
- Rifles
- Espingardas

#### Armas Brancas

- Facas
- Canivetes
- Objetos pontiagudos

#### Outros Objetos

- Bastões
- Objetos contundentes
- Materiais explosivos

## Técnico

### Requisitos de Hardware

#### GPU

- NVIDIA T4 16GB (recomendado)
- CUDA 11.8+
- 16GB RAM

#### CPU

- 8+ cores
- 32GB RAM
- SSD para cache

### Problemas Comuns

#### Erro CUDA

**Problema**: `CUDA not available`

**Solução**:

```bash
nvidia-smi
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

#### Memória Insuficiente

**Problema**: `CUDA out of memory`

**Solução**:

- Reduza o tamanho do batch
- Diminua a resolução
- Ajuste `GPU_MEMORY_FRACTION` no `.env`

## Performance

### Como melhorar a performance?

#### GPU

- Use batch processing
- Ative half precision
- Otimize o cache de modelos

#### CPU

- Ative multiprocessing
- Use vetorização NumPy
- Implemente cache de resultados

### Configurações Recomendadas

```plaintext
# GPU T4
GPU_MEMORY_FRACTION=0.9
BATCH_SIZE=16
USE_HALF_PRECISION=true

# CPU
MAX_WORKERS=8
CACHE_SIZE=1000
USE_MULTIPROCESSING=true
```

## Deployment

### Como fazer deploy no Hugging Face?

1. Configure as credenciais:

    ```bash
    cp .env.example .env.huggingface
    ```

2. Edite as variáveis:

    ```plaintext
    HF_SPACE_ID=seu-espaco
    HF_TOKEN=seu_token
    ```

3. Execute o deploy:

    ```bash
    ./deploy.sh
    ```

### Monitoramento

- Use os logs em `logs/app.log`
- Monitore GPU com `nvidia-smi`
- Verifique métricas no Hugging Face

## Segurança

### Como proteger as credenciais?

1. Use variáveis de ambiente:

    ```bash
    cp .env.example .env
    ```

2. Nunca comite arquivos `.env`
3. Use secrets no Hugging Face

### Validação de Entrada

- Limite o tamanho dos vídeos
- Verifique formatos permitidos
- Sanitize inputs 