# 🚨 Sistema de Detecção de Riscos em Vídeos

Sistema inteligente para detecção de objetos perigosos em vídeos usando IA, desenvolvido para aumentar a segurança em ambientes monitorados.

## 🎯 Funcionalidades

- **Detecção em Tempo Real**: Análise frame a frame de vídeos para identificação de objetos perigosos
- **Alta Precisão**: Utiliza o modelo OWL-ViT (patch16) para detecção precisa de objetos
- **Visualização Clara**: Bounding boxes com scores de confiança em cada detecção
- **Métricas Detalhadas**: Informações completas sobre performance e detecções
- **Interface Web**: Interface amigável usando Gradio para upload e análise de vídeos
- **API REST**: Endpoints para integração com outros sistemas

## 🛠️ Tecnologias Utilizadas

- Python 3.8+
- PyTorch com MPS/CPU
- OWL-ViT (Vision Transformer)
- OpenCV para processamento de vídeo
- FFmpeg para geração de vídeo
- Flask para API REST
- Gradio para interface web

## 📋 Pré-requisitos

1. Python 3.8 ou superior
2. FFmpeg instalado no sistema
3. Pip (gerenciador de pacotes Python)
4. Git

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

## 💻 Uso

### Interface Web
1. Inicie o servidor:
```bash
python app.py
```
2. Acesse `http://localhost:7860` no navegador
3. Faça upload do vídeo e ajuste os parâmetros
4. Clique em "Analisar Vídeo"

### API REST
Endpoints disponíveis:
- `POST /process_video`: Processa um vídeo
- `GET /metrics`: Obtém métricas do último processamento

Exemplo de uso com curl:
```bash
curl -X POST -F "video=@seu_video.mp4" http://localhost:5000/process_video
```

## 📊 Métricas e Parâmetros

### Parâmetros Ajustáveis
- **Threshold**: 0.1 a 1.0 (padrão: 0.4)
- **FPS**: 1 a 30 (padrão: 2)

### Métricas Disponíveis
- Performance (tempo total, FPS, etc.)
- Detecções (quantidade, tipos, confiança)
- Pré-processamento (tamanho, brilho, contraste)
- Uso de memória

## 🔍 Detecções

O sistema detecta:
- Armas brancas (facas, lâminas, etc.)
- Objetos pontiagudos
- Armas de fogo
- Outros objetos perigosos

Cada detecção inclui:
- Bounding box
- Score de confiança
- Timestamp
- Tipo de objeto

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Notas de Versão

### v1.0.0
- Detecção de objetos perigosos
- Interface web Gradio
- API REST
- Métricas detalhadas
- Suporte a MPS/CPU

## ⚠️ Limitações Conhecidas

- Processamento pode ser lento em CPUs menos potentes
- Requer pelo menos 4GB de RAM
- Alguns falsos positivos em condições de baixa luz
- Vídeos muito longos podem consumir bastante memória

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📧 Contato

Para questões e suporte: [SEU_EMAIL]

---
Desenvolvido com ❤️ para o Hackathon FIAP