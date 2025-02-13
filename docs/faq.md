# FAQ

## Geral

### Como o sistema funciona?

O sistema utiliza um modelo de IA (OWL-ViT) para detectar objetos de risco em vídeos.
O processamento pode ser feito em GPU ou CPU, com otimizações específicas para cada caso.

### O que é o OWL-ViT?

O OWL-ViT (Vision Transformer for Open-World Localization) é um modelo de IA que:

- Usa arquitetura Transformer para processar imagens
- Permite detecção zero-shot de objetos
- Aceita queries em linguagem natural
- Não precisa de treinamento específico para novos objetos

### Problemas Conhecidos com OWL-ViT

#### Limitações do Modelo Ensemble

O modelo `owlv2-base-patch16-ensemble` apresenta incompatibilidades com processamento GPU:

- Conflitos com versões estáveis do Transformers
- Problemas de memória em GPUs com menos de 16GB
- Instabilidade em batch processing

**Solução Implementada:**

1. Mudança para modelo base: `owlv2-base-patch16`
2. Atualização do Transformers para branch de desenvolvimento:

    ```bash
    pip install git+https://github.com/huggingface/transformers.git
    ```

3. Ajustes nas configurações de memória GPU:

```python
model = model.to(device='cuda', dtype=torch.float16)
```

#### Comparação de Versões

1. **Modelo Base**
   - Mais estável
   - Menor consumo de memória
   - Compatível com mais GPUs

2. **Modelo Ensemble**
   - Maior precisão
   - Requer mais recursos
   - Melhor para CPU

### Como fazer queries efetivas para o OWL-ViT?

Para melhores resultados, use estas técnicas:

1. **Seja Específico**
   - Bom: "uma pistola preta"
   - Ruim: "arma"

2. **Use Variações**
   - "uma arma de fogo"
   - "uma pistola"
   - "um revólver"

3. **Inclua Contexto**
   - "uma faca na mão de alguém"
   - "uma arma apontada"

4. **Descreva Características**
   - "uma faca com lâmina metálica"
   - "um rifle com coronha"

### Tipos de Objetos Detectados

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

#### Especificações GPU

- NVIDIA T4 16GB (recomendado)
- CUDA 11.8+
- 16GB RAM

#### Especificações CPU

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

### Otimizações de Sistema

#### Ajustes GPU

- Use batch processing
- Ative half precision
- Otimize o cache de modelos

#### Ajustes CPU

- Ative multiprocessing
- Use vetorização NumPy
- Implemente cache de resultados

### Configurações Recomendadas

```plaintext
// Configurações para GPU T4
GPU_MEMORY_FRACTION=0.9
BATCH_SIZE=16
USE_HALF_PRECISION=true

// Configurações para CPU
MAX_WORKERS=8
CACHE_SIZE=1000
USE_MULTIPROCESSING=true
```

## Deployment

### Processo de Deploy no Hugging Face

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

### Sistema de Monitoramento

- Use os logs em `logs/app.log`
- Monitore GPU com `nvidia-smi`
- Verifique métricas no Hugging Face

## Segurança

### Gerenciamento de Credenciais

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

```text
HUGGINGFACE_TOKEN=seu_token
GPU_MEMORY_FRACTION=0.9
MAX_CONCURRENT_REQUESTS=2
```

```text
USE_GPU=true
GPU_DEVICE=0
```

```text
USE_GPU=false
```

```text
HF_SPACE_ID=seu-espaco
HF_TOKEN=seu_token
```
