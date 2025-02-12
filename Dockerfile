FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Configurar ambiente não interativo
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code

# Instalar Python e dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Inicializar Git LFS
RUN git lfs install

# Copiar requirements primeiro para aproveitar cache
COPY requirements.txt .

# Configurar ambiente Python
ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Instalar dependências Python otimizadas para GPU
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Criar diretório de vídeos e cache
RUN mkdir -p /code/videos /code/.cache/huggingface /code/.cache/torch && \
    chmod -R 777 /code/.cache /code/videos

# Configurar variáveis de ambiente
ENV HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code \
    TRANSFORMERS_CACHE=/code/.cache/huggingface \
    TORCH_HOME=/code/.cache/torch \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    SYSTEM=spaces \
    CUDA_VISIBLE_DEVICES=0 \
    HUGGINGFACE_HUB_CACHE=/code/.cache/huggingface \
    HF_HOME=/code/.cache/huggingface \
    TORCH_CUDA_ARCH_LIST="7.5" \
    MAX_WORKERS=2

# Copiar arquivos do projeto
COPY . .

# Expor porta
EXPOSE 7860

# Comando para iniciar a aplicação
CMD ["python3", "app.py"]
