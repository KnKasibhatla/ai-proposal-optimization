#!/bin/bash

# Secure GitHub Push Script - Token handled securely

echo "🚀 Secure GitHub Push Script"
echo "============================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Git not initialized. Run 'git init' first."
    exit 1
fi

# Repository details
repo_url="https://github.com/KnKasibhatla/ai-proposal-optimization.git"
echo "📝 Repository: $repo_url"
echo ""

# Get token securely
read -s -p "🔑 Enter your GitHub Personal Access Token: " github_token
echo ""
echo ""

if [ -z "$github_token" ]; then
    echo "❌ No token provided. Cannot proceed."
    exit 1
fi

# Check if repository exists
echo "📝 Checking repository access..."
if ! curl -s -H "Authorization: token $github_token" https://api.github.com/repos/KnKasibhatla/ai-proposal-optimization | grep -q '"name"'; then
    echo "❌ Cannot access repository. Check your token permissions."
    exit 1
fi

echo "✅ Repository access confirmed!"
echo ""

# Update remote URL
echo "🔗 Updating remote repository..."
git remote set-url origin "$repo_url"

# Push code
echo "📤 Pushing code to GitHub..."
echo ""

# Try to push
if git push https://KnKasibhatla:$github_token@github.com/KnKasibhatla/ai-proposal-optimization.git main; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🎉 Your code is now backed up at:"
    echo "   https://github.com/KnKasibhatla/ai-proposal-optimization"
    echo ""
else
    echo ""
    echo "❌ Push failed. Please check:"
    echo "   1. Token has 'repo' scope permissions"
    echo "   2. Repository exists and you have access"
    echo "   3. Token is not expired"
fi

# Clear token from memory
unset github_token