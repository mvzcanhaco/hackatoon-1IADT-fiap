---
title: Weapon Detection App
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.15.0
app_file: app.py
pinned: false
license: mit
hardware: true
resources:
  accelerator: T4
  gpu: true
---

# Sistema de Detecção de Riscos em Vídeo

Este projeto implementa um sistema de detecção de riscos em vídeo utilizando YOLOv8 e Clean Architecture.

## Pré-requisitos

- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)
- Ambiente virtual Python (recomendado)

## Configuração do Ambiente

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OU
.venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
```
NOTIFICATION_API_KEY=sua_chave_api
```

## Executando o Projeto

1. Ative o ambiente virtual (se ainda não estiver ativo)

2. Execute o aplicativo:
```bash
python src/main.py
```

3. Acesse a interface web através do navegador no endereço mostrado no terminal (geralmente http://localhost:7860)

## Funcionalidades

- Upload de vídeos para análise
- Detecção de objetos em tempo real
- Configuração de parâmetros de detecção
- Sistema de notificações
- Monitoramento de recursos do sistema

## Estrutura do Projeto

O projeto segue os princípios da Clean Architecture:

- `domain/`: Regras de negócio e entidades
- `application/`: Casos de uso e interfaces
- `infrastructure/`: Implementações concretas
- `presentation/`: Interface com usuário (Gradio)

## Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Tecnologias

- Python 3.8+
- PyTorch com CUDA
- OWL-ViT
- Gradio
- FFmpeg

## Requisitos de Hardware

- GPU NVIDIA T4 (fornecida pelo Hugging Face)
- 16GB de RAM
- Armazenamento para cache de modelos

## Limitações

- Processamento pode ser lento em CPUs menos potentes
- Requer GPU para melhor performance
- Alguns falsos positivos em condições de baixa luz

---
Desenvolvido com ❤️ para o Hackathon FIAP
