# Detecção de Armas - FIAP Vision Guard - Hackatoon 1IADT

## Sobre o Projeto

A FIAP VisionGuard é uma empresa especializada em monitoramento de câmeras de segurança que busca inovar através da implementação de tecnologias avançadas de detecção de riscos. Este projeto demonstra a viabilidade de uma nova funcionalidade que utiliza Inteligência Artificial para identificar objetos potencialmente perigosos em tempo real, como armas brancas (facas, tesouras e similares) e outros objetos de risco.

### Objetivo

O sistema visa otimizar a segurança de estabelecimentos e comércios através de:

- Detecção automática de objetos perigosos
- Emissão de alertas em tempo real para centrais de segurança
- Análise contínua de feeds de vídeo
- Redução do tempo de resposta a incidentes

Sistema de detecção de objetos de risco em vídeos usando OWLV2-ViT e processamento
GPU/CPU otimizado.

[![Open in Hugging Face][hf-badge]][hf-space]
[![GitHub][gh-badge]][gh-repo]

[hf-badge]: https://img.shields.io/badge/Hugging%20Face-Spaces-yellow
[hf-space]: https://huggingface.co/spaces/marcuscanhaco/weapon-detection-app
[gh-badge]: https://img.shields.io/badge/GitHub-Repo-blue
[gh-repo]: https://github.com/mvzcanhaco/hackatoon-1IADT-fiap

## Funcionalidades

- Detecção de objetos de risco em vídeos
- Processamento otimizado em GPU (NVIDIA T4) e CPU
- Interface web intuitiva com Gradio
- API REST para integração
- Suporte a webhooks para notificações
- Métricas detalhadas de processamento

## Requisitos

- Python 3.10+
- CUDA 11.8+ (para GPU)
- NVIDIA T4 16GB ou superior (recomendado)
- 16GB RAM mínimo

## Instalação

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

## Uso

1. Inicie a aplicação:

    ```bash
    python app.py
    ```

2. Acesse: `http://localhost:7860`

3. Upload de vídeo:

    - Arraste ou selecione um vídeo
    - Ajuste as configurações
    - Clique em "Detectar"

## Documentação

- [Arquitetura do Sistema](docs/architecture/overview.md)
- [Instalação e Configuração](docs/setup/installation.md)
- [API e Interface](docs/api/interface.md)

## Arquitetura

O projeto segue os princípios da Clean Architecture:

```plaintext
src/
├── domain/         # Regras de negócio
├── application/    # Casos de uso
├── infrastructure/ # Implementações
└── presentation/   # Interface
```

[Detalhes da arquitetura](docs/architecture/overview.md)

## Deploy no Hugging Face

1. Configure as credenciais:

    ```bash
    cp .env.example .env.huggingface
    ```

2. Execute o deploy:

    ```bash
    ./deploy.sh
    ```

## Máquinas Recomendadas

### GPU

- NVIDIA T4 16GB (Hugging Face Pro)
- NVIDIA A100 (Performance máxima)
- NVIDIA V100 (Alternativa)

### CPU

- 8+ cores
- 32GB+ RAM
- SSD para armazenamento

## Interface

### Componentes

- Upload de vídeo (MP4, AVI, MOV)
- Configurações de detecção
- Visualização de resultados
- Métricas em tempo real

## Links

- [Hugging Face Space][hf-space]
- [GitHub Repository][gh-repo]
- [Documentação](docs/)
- [Issues](https://github.com/seu-usuario/hackatoon-1iadt/issues)

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE)
para detalhes.

## Contribuição

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Suporte

- Abra uma [issue](https://github.com/seu-usuario/hackatoon-1iadt/issues)
- Consulte a [documentação](docs/)
- Entre em contato com a equipe
