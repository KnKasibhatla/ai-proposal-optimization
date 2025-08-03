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

# Pre-configured GitHub repository details
repo_url="https://github.com/KnKasibhatla/ai-proposal-optimization.git"

# Get token from environment variable or prompt user
if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN environment variable not set."
    echo ""
    read -s -p "🔑 Enter your GitHub Personal Access Token: " github_token
    echo ""
else
    github_token="$GITHUB_TOKEN"
fi

if [ -z "$github_token" ]; then
    echo "❌ No token provided. Cannot proceed."
    exit 1
fi

echo "📝 Checking if GitHub repository exists..."

# Check if repository exists
if ! curl -s -H "Authorization: token $github_token" https://api.github.com/repos/KnKasibhatla/ai-proposal-optimization | grep -q '"name"'; then
    echo ""
    echo "❌ Repository doesn't exist yet!"
    echo ""
    echo "🔧 Please create the repository first:"
    echo "   1. Go to: https://github.com/new"
    echo "   2. Repository name: AIProposalPricing"
    echo "   3. Description: AI-powered bid optimization platform"
    echo "   4. Make it Public or Private (your choice)"
    echo "   5. ❗ DO NOT initialize with README, .gitignore, or license"
    echo "   6. Click 'Create repository'"
    echo ""
    echo "Then run this script again: ./push_to_github.sh"
    exit 1
fi

echo "✅ Repository exists! Proceeding with push..."
echo "   Repository: $repo_url"
echo ""

echo ""
echo "🔗 Adding remote repository..."
git remote add origin "$repo_url" 2>/dev/null || {
    echo "⚠️  Remote 'origin' already exists. Updating URL..."
    git remote set-url origin "$repo_url"
}

echo "📤 Pushing code to GitHub..."
echo ""

# Configure Git credential helper for token authentication
git config credential.helper store
echo "https://KnKasibhatla:$github_token@github.com" > ~/.git-credentials

# Push to main branch
echo "🔐 Authenticating with GitHub token..."
git push -u origin main 2>/dev/null || {
    # If main doesn't work, try master
    echo "Trying branch 'master'..."
    git branch -M main
    git push -u origin main 2>/dev/null || {
        echo "❌ Push failed. Please check your token and repository access."
        echo ""
        echo "🔐 Token troubleshooting:"
        echo "   1. Verify your token has 'repo' scope"
        echo "   2. Check if repository exists at: $repo_url"
        echo "   3. Ensure token is not expired"
        echo ""
        # Clean up credentials on failure
        rm -f ~/.git-credentials
        exit 1
    }
}

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
# Clean up credentials after successful push
rm -f ~/.git-credentials

echo "🎉 Your code is now backed up on GitHub at:"
echo "   https://github.com/KnKasibhatla/ai-proposal-optimization"
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