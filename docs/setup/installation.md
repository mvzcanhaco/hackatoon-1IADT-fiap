# Instalação e Configuração

## Requisitos do Sistema

### Hardware Recomendado
- **GPU**: NVIDIA T4 16GB ou superior
- **CPU**: 4+ cores
- **RAM**: 16GB mínimo
- **Armazenamento**: 10GB+ disponível

### Software Necessário
- Python 3.10+
- CUDA 11.8+ (para GPU)
- Git

## Instalação Local

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/hackatoon-1iadt.git
cd hackatoon-1iadt
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas configurações:
```
HUGGINGFACE_TOKEN=seu_token
GPU_MEMORY_FRACTION=0.9
MAX_CONCURRENT_REQUESTS=2
```

## Configuração do Ambiente

### GPU (NVIDIA)

1. Verifique a instalação do CUDA:
```bash
nvidia-smi
```

2. Ajuste as configurações de GPU em `.env`:
```
USE_GPU=true
GPU_DEVICE=0
```

### CPU

Para usar apenas CPU:
```
USE_GPU=false
```

## Deployment no Hugging Face

1. Configure as variáveis de ambiente do Hugging Face:
```bash
cp .env.example .env.huggingface
```

2. Edite `.env.huggingface`:
```
HF_SPACE_ID=seu-espaco
HF_TOKEN=seu_token
```

3. Execute o deploy:
```bash
./deploy.sh
```

## Verificação da Instalação

1. Execute os testes:
```bash
pytest
```

2. Inicie a aplicação:
```bash
python app.py
```

3. Acesse: http://localhost:7860

## Troubleshooting

### Problemas Comuns

1. **Erro CUDA**
   - Verifique a instalação do CUDA
   - Confirme compatibilidade de versões

2. **Memória Insuficiente**
   - Ajuste `GPU_MEMORY_FRACTION`
   - Reduza `MAX_CONCURRENT_REQUESTS`

3. **Falha no Deploy**
   - Verifique credenciais do Hugging Face
   - Confirme permissões do espaço

### Logs

- Logs da aplicação: `logs/app.log`
- Logs do GPU: `logs/gpu.log`

## Suporte

Para problemas e dúvidas:
1. Abra uma issue no GitHub
2. Consulte a documentação completa
3. Entre em contato com a equipe de suporte 