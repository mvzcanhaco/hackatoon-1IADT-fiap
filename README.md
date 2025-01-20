# Processador de Vídeo em Tempo Real

Esta aplicação Streamlit permite processar vídeos em tempo real, seja através de upload de arquivo ou webcam.

## Funcionalidades

- Upload de vídeos (MP4, MOV, AVI)
- Captura e processamento de vídeo via webcam
- Processamento em tempo real (conversão para escala de cinza)
- Interface intuitiva com visualização lado a lado (original vs processado)

## Requisitos

As dependências necessárias estão listadas no arquivo `requirements.txt`. Para instalá-las, execute:

```bash
pip install -r requirements.txt
```

## Executando Localmente

1. Clone este repositório:
```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DO_DIRETORIO]
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run app.py
```

4. Acesse a aplicação em seu navegador (geralmente em http://localhost:8501)

## Implantação no Hugging Face Spaces

1. Crie um novo Space no Hugging Face:
   - Acesse https://huggingface.co/spaces
   - Clique em "Create new Space"
   - Selecione "Streamlit" como SDK
   - Escolha um nome para seu Space

2. Configure seu repositório Git:
```bash
git init
git add .
git commit -m "Primeira versão"
git branch -M main
git remote add origin https://huggingface.co/spaces/[SEU_USUARIO]/[NOME_DO_SPACE]
git push -u origin main
```

## Estrutura do Projeto

```
.
├── app.py              # Aplicação principal
├── requirements.txt    # Dependências do projeto
└── README.md          # Documentação
```

## Notas Importantes

- A aplicação usa `opencv-python-headless` para compatibilidade com Hugging Face Spaces
- O processamento de vídeo atual é um exemplo simples (escala de cinza)
- Para adicionar mais funcionalidades de processamento, modifique a função `process_frame` em `app.py` 