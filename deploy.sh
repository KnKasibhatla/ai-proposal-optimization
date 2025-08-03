#!/bin/bash

# AWS Deployment Script for AI Bid Optimization Platform
# Run this script to deploy your application to AWS Elastic Beanstalk

echo "🚀 Starting AWS Deployment Process..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install it first:"
    echo "pip install awscli"
    exit 1
fi

# Check if EB CLI is installed
if ! command -v eb &> /dev/null; then
    echo "❌ Elastic Beanstalk CLI not found. Please install it first:"
    echo "pip install awsebcli"
    exit 1
fi

# Check AWS credentials
echo "🔐 Checking AWS credentials..."
aws sts get-caller-identity > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ AWS credentials not configured. Please run:"
    echo "aws configure"
    exit 1
fi

echo "✅ AWS credentials verified"

# Prepare deployment files
echo "📦 Preparing deployment files..."
cp requirements_aws.txt requirements.txt

# Check if EB is already initialized
if [ ! -f .elasticbeanstalk/config.yml ]; then
    echo "🆕 Initializing Elastic Beanstalk application..."
    echo "Please follow the prompts to configure your application:"
    echo "- Choose your preferred region"
    echo "- Select 'Create New Application'"
    echo "- Name it 'ai-bid-optimization'"
    echo "- Choose Python 3.9 platform"
    eb init
else
    echo "✅ Elastic Beanstalk already initialized"
fi

# Ask user about environment creation
read -p "🤔 Do you want to create a new environment? (y/n): " create_env

if [ "$create_env" = "y" ] || [ "$create_env" = "Y" ]; then
    echo "🌍 Creating production environment..."
    echo "This may take 5-10 minutes..."
    eb create ai-bid-prod --instance-type t3.micro
else
    echo "📤 Deploying to existing environment..."
    eb deploy
fi

# Show deployment status
echo "📊 Checking deployment status..."
eb status

# Show application URL
echo ""
echo "🎉 Deployment completed!"
echo "📍 Your application should be available at the URL shown above"
echo ""
echo "📋 Useful commands:"
echo "  eb status    - Check application status"
echo "  eb logs      - View application logs"
echo "  eb deploy    - Deploy updates"
echo "  eb terminate - Terminate environment (when done)"
echo ""
echo "💡 Next steps:"
echo "1. Test your application using the provided URL"
echo "2. Configure a custom domain (optional)"
echo "3. Enable HTTPS/SSL certificate"
echo "4. Set up monitoring and alerts"

echo ""
echo "✅ Deployment script completed!"