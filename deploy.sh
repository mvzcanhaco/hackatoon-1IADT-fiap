#!/bin/bash

echo "🚀 Iniciando processo de deploy..."

# Solicitar mensagem do commit
echo "📝 Digite a mensagem do commit (ou pressione Enter para mensagem padrão):"
read commit_message

# Se nenhuma mensagem foi fornecida, usar mensagem padrão
if [ -z "$commit_message" ]; then
    commit_message="feat: atualização do detector com otimizações para GPU T4"
fi

# Deploy para GitHub
echo "🚀 Preparando deploy para GitHub..."

# Adicionar alterações exceto vídeos e arquivos grandes
echo "📦 Adicionando arquivos para GitHub..."
git add .

# Fazer commit
git commit -m "$commit_message"

# Force push para GitHub
echo "🚀 Forçando push para GitHub..."
git push -f origin main

if [ $? -eq 0 ]; then
    echo "✅ Deploy para GitHub concluído com sucesso!"
else
    echo "❌ Erro durante o deploy para GitHub"
    exit 1
fi

# Deploy para Hugging Face
echo "🚀 Preparando deploy para Hugging Face..."

# Verificar se o remote do Hugging Face existe
if ! git remote | grep -q "^space$"; then
    echo "❌ Remote 'space' não encontrado!"
    echo "⚠️ Execute os seguintes comandos:"
    echo "   git remote add space https://huggingface.co/spaces/SEU_USUARIO/NOME_DO_SPACE"
    exit 1
fi

# Adicionar todos os arquivos incluindo vídeos
echo "📦 Adicionando todos os arquivos..."
git add --all

# Fazer commit
git commit -m "$commit_message"

# Force push para Hugging Face
echo "🚀 Enviando para Hugging Face Space..."
git push -f space main

if [ $? -eq 0 ]; then
    echo "✅ Deploy para Hugging Face concluído com sucesso!"
    echo "🌐 Seu app estará disponível em alguns minutos em:"
    echo "   https://huggingface.co/spaces/marcuscanhaco/weapon-detection-app"
    echo ""
    echo "⚠️ Lembre-se de verificar no Hugging Face Space se:"
    echo "  1. O Space está configurado para usar GPU T4"
    echo "  2. As variáveis de ambiente estão configuradas corretamente:"
    echo "     - HUGGING_FACE_TOKEN"
    echo "     - NOTIFICATION_EMAIL"
    echo "     - SENDGRID_API_KEY"
    echo "  3. Os requisitos de memória estão adequados"
else
    echo "❌ Erro durante o deploy para Hugging Face"
    exit 1
fi 