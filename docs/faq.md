# FAQ

## Geral

### Como o sistema funciona?

O sistema utiliza um modelo de IA (OWL-ViT) para detectar objetos de risco em vídeos.
O processamento é feito frame a frame em GPU ou CPU, com otimizações específicas para cada caso.

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

#### Problemas com Batch Processing

O processamento em batch apresenta instabilidades conhecidas:

1. **Erros de Shape**
   ```
   ERROR: shape '[4, 21, 512]' is invalid for input of size 44544
   ERROR: shape '[2, 43, 512]' is invalid for input of size 44544
   ```

2. **Causas Identificadas**
   - Inconsistência no padding de imagens em batch
   - Variações no tamanho dos tensores de entrada
   - Incompatibilidade com certas configurações de GPU

3. **Solução Recomendada**
   ```python
   # Processamento seguro frame a frame
   batch_size = 1  # Processa um frame por vez
   ```

4. **Benefícios do Processamento Individual**
   - Maior estabilidade
   - Melhor gerenciamento de memória
   - Resultados mais consistentes
   - Facilidade de debug
   - Menor chance de OOM (Out of Memory)

5. **Trade-offs**
   - Performance levemente reduzida
   - Processamento mais serializado
   - Maior tempo total de execução

#### Comparação de Abordagens

| Aspecto | Batch Processing | Frame a Frame |
|---------|------------------|---------------|
| Velocidade | Mais rápido (quando funciona) | Mais lento |
| Estabilidade | Baixa | Alta |
| Uso de Memória | Alto/Imprevisível | Baixo/Consistente |
| Confiabilidade | Baixa | Alta |
| Debug | Difícil | Fácil |

#### Recomendações de Uso

1. **Produção**
   ```python
   # Configuração recomendada para produção
   batch_size = 1
   resolution = 640
   fps = 2
   ```

2. **Desenvolvimento**
   ```python
   # Configuração para testes
   batch_size = 1
   resolution = 480
   fps = 1
   ```

3. **Monitoramento**
   ```python
   # Log de progresso a cada 10 frames
   if i % 10 == 0:
       logger.info(f"Processados {i}/{len(frames)} frames")
   ```

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

- Use processamento frame a frame (padrão)
- Diminua a resolução se necessário
- Ajuste `GPU_MEMORY_FRACTION` no `.env`

## Performance

### Otimizações de Sistema

#### Ajustes GPU

- Processamento frame a frame otimizado
- Ative half precision
- Otimize o cache de modelos e frames
- Gerenciamento eficiente de memória

#### Ajustes CPU

- Processamento sequencial otimizado
- Use vetorização NumPy
- Implemente cache de resultados

### Configurações Recomendadas

```plaintext
// Configurações para GPU T4
GPU_MEMORY_FRACTION=0.9
BATCH_SIZE=1  # Processamento frame a frame
USE_HALF_PRECISION=true

// Configurações para CPU
MAX_WORKERS=8
CACHE_SIZE=1000
USE_MULTIPROCESSING=true
```

### Sistema de Monitoramento

- Use os logs em `logs/app.log` para acompanhar o processamento frame a frame
- Monitore GPU com `nvidia-smi`
- Verifique métricas no Hugging Face
- Acompanhe logs de progresso a cada 10 frames

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
