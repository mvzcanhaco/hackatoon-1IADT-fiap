#!/bin/bash

echo "🚀 Iniciando processo de deploy..."



# Verificar e criar diretórios necessários
echo "📁 Verificando estrutura de diretórios..."
mkdir -p videos/seguro videos/risco_detectado

# Solicitar mensagem do commit
echo "📝 Digite a mensagem do commit (ou pressione Enter para mensagem padrão):"
read commit_message

# Se nenhuma mensagem foi fornecida, usar mensagem padrão
if [ -z "$commit_message" ]; then
    commit_message="feat: atualização do detector com otimizações para GPU T4"
fi

# Deploy para GitHub
echo "🚀 Preparando deploy para GitHub..."

# Salvar estado atual dos vídeos
echo "📦 Salvando estado dos vídeos..."
git stash push videos/

# Adicionar alterações exceto vídeos
echo "📦 Adicionando arquivos para GitHub..."
git add .

# Verificar se há alterações para commitar
if [[ -n $(git status -s) ]]; then
    echo "📝 Existem alterações para commitar no GitHub"
    
    # Fazer commit
    git commit -m "$commit_message"
    
    # Push para GitHub
    echo "🚀 Enviando para GitHub..."
    git push origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ Deploy para GitHub concluído com sucesso!"
    else
        echo "❌ Erro durante o deploy para GitHub"
        git stash pop  # Restaurar vídeos
        exit 1
    fi
else
    echo "✨ Workspace limpo, nenhuma alteração para GitHub"
fi

# Restaurar vídeos
echo "📦 Restaurando vídeos..."
git stash pop

# Deploy para Hugging Face
echo "🚀 Preparando deploy para Hugging Face..."

# Verificar arquivos grandes
echo "🔍 Verificando arquivos grandes..."
find . -size +100M -not -path "*.git*" | while read file; do
    echo "⚠️ Arquivo grande encontrado: $file"
    echo "Verificando se está configurado no Git LFS..."
    if ! git check-attr filter "$file" | grep -q "lfs"; then
        echo "❌ $file não está configurado no Git LFS!"
        exit 1
    fi
done

# Adicionar todos os arquivos incluindo vídeos
echo "📦 Adicionando todos os arquivos incluindo vídeos..."
git add --all

# Verificar se há alterações para o Hugging Face
if [[ -n $(git status -s) ]]; then
    echo "📝 Existem alterações para commitar no Hugging Face"
    
    # Fazer commit
    git commit -m "$commit_message"
    
    
    echo "🚀 Enviando para Hugging Face Space..."
    git push space main
    
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
else
    echo "✨ Workspace limpo, nenhuma alteração para Hugging Face"
fi 