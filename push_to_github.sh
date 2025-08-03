#!/bin/bash

# Script to push your code to GitHub

echo "🚀 GitHub Repository Setup Script"
echo "================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Git not initialized. Run 'git init' first."
    exit 1
fi

echo "📝 Follow these steps to push your code to GitHub:"
echo ""
echo "1️⃣  Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name it: ai-bid-optimization"
echo "   - Make it Private (recommended) or Public"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2️⃣  After creating the repo, GitHub will show you commands."
echo "   Copy the repository URL (looks like: https://github.com/yourusername/ai-bid-optimization.git)"
echo ""
read -p "3️⃣  Enter your GitHub repository URL: " repo_url

if [ -z "$repo_url" ]; then
    echo "❌ No URL provided. Exiting."
    exit 1
fi

echo ""
echo "🔗 Adding remote repository..."
git remote add origin "$repo_url" 2>/dev/null || {
    echo "⚠️  Remote 'origin' already exists. Updating URL..."
    git remote set-url origin "$repo_url"
}

echo "📤 Pushing code to GitHub..."
echo ""

# Push to main branch
git push -u origin main 2>/dev/null || {
    # If main doesn't work, try master
    echo "Trying branch 'master'..."
    git branch -M main
    git push -u origin main 2>/dev/null || {
        echo "❌ Push failed. You may need to authenticate."
        echo ""
        echo "🔐 If you see an authentication error:"
        echo "   1. Make sure you're logged into GitHub"
        echo "   2. For HTTPS: Use a Personal Access Token (not password)"
        echo "      - Go to GitHub Settings > Developer Settings > Personal Access Tokens"
        echo "      - Generate a new token with 'repo' scope"
        echo "      - Use the token as your password"
        echo ""
        echo "   3. For SSH: Set up SSH keys"
        echo "      - Run: ssh-keygen -t ed25519 -C 'your_email@example.com'"
        echo "      - Add the public key to GitHub Settings > SSH Keys"
        echo "      - Change remote to SSH: git remote set-url origin git@github.com:username/repo.git"
        echo ""
        echo "📝 Then run: git push -u origin main"
        exit 1
    }
}

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "🎉 Your code is now backed up on GitHub at:"
echo "   $repo_url"
echo ""
echo "📋 Useful Git commands:"
echo "   git status           - Check what's changed"
echo "   git add .            - Stage all changes"
echo "   git commit -m 'msg'  - Commit changes"
echo "   git push             - Push to GitHub"
echo "   git pull             - Pull latest from GitHub"
echo ""
echo "🔄 To push future changes:"
echo "   1. git add ."
echo "   2. git commit -m 'Your message'"
echo "   3. git push"
echo ""