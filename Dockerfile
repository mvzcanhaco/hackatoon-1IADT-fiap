FROM python:3.8-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos do projeto
COPY requirements.txt .
COPY *.py .
COPY .env.example .env

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Porta para a aplicação
EXPOSE 7860

# Comando para iniciar a aplicação
CMD ["python", "app_hf.py"]
