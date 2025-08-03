# 🚀 AWS Deployment Guide for AI Bid Optimization Platform

## Prerequisites

1. **AWS Account**: Make sure you have an active AWS account
2. **AWS CLI**: Install and configure AWS CLI
3. **EB CLI**: Install Elastic Beanstalk CLI

## Step 1: Install Required Tools

```bash
# Install AWS CLI (if not already installed)
pip install awscli

# Configure AWS CLI with your credentials
aws configure

# Install Elastic Beanstalk CLI
pip install awsebcli
```

## Step 2: Prepare Your Application

```bash
# Navigate to your project directory
cd /Users/nkasibhatla/WorkingPredictor

# Copy requirements for AWS
cp requirements_aws.txt requirements.txt
```

## Step 3: Initialize Elastic Beanstalk

```bash
# Initialize EB application
eb init

# When prompted:
# - Select your region (e.g., us-west-2)
# - Choose "Create New Application" 
# - Name: "ai-bid-optimization"
# - Platform: Python 3.9
# - Use CodeCommit: No
# - Setup SSH: Yes (recommended)
```

## Step 4: Create Environment

```bash
# Create production environment
eb create ai-bid-prod

# This will:
# - Create EC2 instances
# - Set up load balancer
# - Deploy your application
# - Provide a public URL
```

## Step 5: Deploy Updates

```bash
# Deploy any changes
eb deploy

# Check status
eb status

# View logs
eb logs
```

## Step 6: Access Your Application

After deployment, EB will provide a URL like:
`http://ai-bid-prod.us-west-2.elasticbeanstalk.com`

## Environment Variables (Optional)

```bash
# Set environment variables if needed
eb setenv DEBUG=False
eb setenv FLASK_ENV=production
```

## Cost Optimization

- **Instance Type**: Start with t3.micro (free tier eligible)
- **Auto Scaling**: Configure based on usage
- **Load Balancer**: Can be disabled for single instance

## Security Considerations

1. **HTTPS**: Enable SSL certificate in EB console
2. **Environment Variables**: Store sensitive data as environment variables
3. **Database**: Consider RDS for production data storage
4. **Backup**: Set up automated backups

## Monitoring

- **CloudWatch**: Monitor application metrics
- **Health Dashboard**: Check application health
- **Logs**: Access via EB CLI or AWS Console

## Alternative Deployment Options

### Option 2: AWS EC2 + Docker
- More control over environment
- Use Docker containers
- Manual setup required

### Option 3: AWS Lambda + API Gateway
- Serverless deployment
- Pay per request
- Good for low traffic

### Option 4: AWS ECS/Fargate
- Container orchestration
- Better for microservices
- More complex setup

## Troubleshooting

Common issues and solutions:

1. **Deployment Fails**: Check eb logs for errors
2. **502 Bad Gateway**: Usually application startup issues
3. **File Uploads**: Ensure upload directories have proper permissions
4. **Dependencies**: Verify all requirements in requirements.txt

## Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificate enabled
- [ ] Database configured (if needed)
- [ ] Monitoring set up
- [ ] Backup strategy implemented
- [ ] Domain name configured (optional)
- [ ] Health checks enabled
- [ ] Auto-scaling configured

## Support

For issues:
1. Check AWS Elastic Beanstalk documentation
2. Review CloudWatch logs
3. Use AWS support (if you have a support plan)