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

# Detector de Riscos em Vídeo

Sistema de detecção de objetos de risco em vídeos usando OWL-ViT e processamento GPU/CPU otimizado.

[![Open in Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/seu-usuario/seu-espaco)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/seu-usuario/hackatoon-1iadt)

## 🚀 Funcionalidades

- Detecção de objetos de risco em vídeos
- Processamento otimizado em GPU (NVIDIA T4) e CPU
- Interface web intuitiva com Gradio
- API REST para integração
- Suporte a webhooks para notificações
- Métricas detalhadas de processamento

## 📋 Requisitos

- Python 3.10+
- CUDA 11.8+ (para GPU)
- NVIDIA T4 16GB ou superior (recomendado)
- 16GB RAM mínimo

## 🔧 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/hackatoon-1iadt.git
cd hackatoon-1iadt
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure o ambiente:
```bash
cp .env.example .env
```

[Documentação completa de instalação](docs/setup/installation.md)

## 💻 Uso

1. Inicie a aplicação:
```bash
python app.py
```

2. Acesse: http://localhost:7860

3. Upload de vídeo:
   - Arraste ou selecione um vídeo
   - Ajuste as configurações
   - Clique em "Detectar"

## 📚 Documentação

- [Arquitetura do Sistema](docs/architecture/overview.md)
- [Instalação e Configuração](docs/setup/installation.md)
- [API e Interface](docs/api/interface.md)

## 🏗️ Arquitetura

O projeto segue os princípios da Clean Architecture:

```
src/
├── domain/         # Regras de negócio
├── application/    # Casos de uso
├── infrastructure/ # Implementações
└── presentation/   # Interface
```

[Detalhes da arquitetura](docs/architecture/overview.md)

## 🚀 Deploy no Hugging Face

1. Configure as credenciais:
```bash
cp .env.example .env.huggingface
```

2. Execute o deploy:
```bash
./deploy.sh
```

[Instruções detalhadas de deploy](docs/setup/installation.md#deployment-no-hugging-face)

## 💪 Máquinas Recomendadas

### GPU
- NVIDIA T4 16GB (Hugging Face Pro)
- NVIDIA A100 (Performance máxima)
- NVIDIA V100 (Alternativa)

### CPU
- 8+ cores
- 32GB+ RAM
- SSD para armazenamento

## 🔍 Interface

### Componentes
- Upload de vídeo (MP4, AVI, MOV)
- Configurações de detecção
- Visualização de resultados
- Métricas em tempo real

[Documentação da interface](docs/api/interface.md)

## 🔗 Links

- [Hugging Face Space](https://huggingface.co/spaces/seu-usuario/seu-espaco)
- [GitHub Repository](https://github.com/seu-usuario/hackatoon-1iadt)
- [Documentação](docs/)
- [Issues](https://github.com/seu-usuario/hackatoon-1iadt/issues)

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Contribuição

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📞 Suporte

- Abra uma [issue](https://github.com/seu-usuario/hackatoon-1iadt/issues)
- Consulte a [documentação](docs/)
- Entre em contato com a equipe

---
Desenvolvido com ❤️ para o Hackathon FIAP
