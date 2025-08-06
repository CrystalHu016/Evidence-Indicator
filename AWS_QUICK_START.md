# 🚀 AWS Quick Start Guide - Evidence Indicator RAG

## 📋 Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** (for ECS/Fargate deployment)
4. **OpenAI API Key** ready

## ⚡ Quick Deployment Options

### Option 1: Lambda (Serverless) - Fastest Setup
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-key-here"

# Deploy to Lambda
./deploy_lambda.sh
```

### Option 2: ECS/Fargate (Containerized) - Production Ready
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-key-here"

# Deploy to ECS/Fargate
./deploy_ecs.sh
```

## 🔧 Setup Steps

### 1. Install AWS CLI
```bash
# macOS
brew install awscli

# Ubuntu/Debian
sudo apt-get install awscli

# Windows
# Download from https://aws.amazon.com/cli/
```

### 2. Configure AWS Credentials
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter your default region (e.g., us-east-1)
# Enter your output format (json)
```

### 3. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-openai-key-here"
```

## 🚀 Deploy

### Lambda Deployment (Recommended for Testing)
```bash
./deploy_lambda.sh
```

**What it does:**
- Creates Lambda function with UltraFastRAG
- Sets up API Gateway for HTTP access
- Configures IAM roles and permissions
- Provides API URL for testing

**Expected Output:**
```
🎉 Deployment completed successfully!
📋 Function Name: evidence-indicator-rag
🌍 Region: us-east-1
🌐 API URL: https://abc123.execute-api.us-east-1.amazonaws.com/prod/query
```

### ECS/Fargate Deployment (Production)
```bash
./deploy_ecs.sh
```

**What it does:**
- Builds and pushes Docker image to ECR
- Creates ECS cluster and service
- Sets up Application Load Balancer (optional)
- Configures auto-scaling and monitoring

## 🧪 Test Your Deployment

### 1. Quick Test
```bash
# Test with curl
curl -X POST https://YOUR_API_URL/query \
  -H "Content-Type: application/json" \
  -d '{"query": "コンバインとは何ですか"}'
```

### 2. Comprehensive Testing
```bash
# Install test dependencies
pip install requests

# Run full test suite
python test_aws_deployment.py https://YOUR_API_URL
```

### 3. Expected Response Format
```json
{
  "answer": "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
  "source_document": "検索ヒットのチャンクを含む文書...",
  "evidence_text": "根拠情報",
  "start_char": 1,
  "end_char": 35,
  "processing_time": 1.25,
  "confidence": 0.85,
  "model": "UltraFastRAG"
}
```

## 📊 Monitor Your Deployment

### Lambda Monitoring
- **CloudWatch Logs**: `/aws/lambda/evidence-indicator-rag`
- **Metrics**: Invocations, Duration, Errors
- **Console**: AWS Lambda Console

### ECS Monitoring
- **CloudWatch Logs**: `/ecs/evidence-indicator-task`
- **Metrics**: CPU, Memory, Network
- **Console**: AWS ECS Console

## 💰 Cost Estimation

### Lambda (Pay-per-use)
- **1,000 queries/month**: ~$0.50-2.00
- **10,000 queries/month**: ~$5.00-20.00
- **100,000 queries/month**: ~$50.00-200.00

### ECS/Fargate (24/7)
- **1 vCPU + 1GB RAM**: ~$30-50/month
- **2 vCPU + 2GB RAM**: ~$60-100/month

## 🔧 Troubleshooting

### Common Issues

1. **Cold Start Latency (Lambda)**
   - Use provisioned concurrency
   - Keep function warm with scheduled invocations

2. **Memory Issues**
   - Increase Lambda memory allocation
   - Monitor ECS memory usage

3. **Timeout Errors**
   - Increase Lambda timeout (max 15 minutes)
   - Optimize RAG processing

4. **API Key Issues**
   - Check Secrets Manager configuration
   - Verify environment variables

### Debug Commands
```bash
# Check Lambda logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/evidence-indicator-rag \
  --start-time $(date -d '1 hour ago' +%s)000

# Check ECS service status
aws ecs describe-services \
  --cluster evidence-indicator-cluster \
  --services evidence-indicator-service

# Test API endpoint
curl -v https://YOUR_API_URL/health
```

## 🎯 Performance Targets

- **Response Time**: <2 seconds (95th percentile)
- **Throughput**: 1000+ requests/minute
- **Availability**: 99.9% uptime
- **Error Rate**: <1%

## 📈 Scaling

### Lambda Auto-scaling
- Automatic scaling based on demand
- No configuration needed

### ECS Auto-scaling
```bash
# Set up auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/evidence-indicator-cluster/evidence-indicator-service \
  --min-capacity 1 \
  --max-capacity 10
```

## 🔐 Security Best Practices

1. **Use Secrets Manager** for API keys
2. **Enable VPC** for private networking
3. **Configure security groups** properly
4. **Enable CloudTrail** for audit logging
5. **Use IAM roles** with minimal permissions

## 📞 Support

- **Documentation**: Check `aws_deployment_guide.md`
- **Issues**: Review CloudWatch logs
- **Performance**: Monitor CloudWatch metrics

## 🎉 Success!

Your Evidence Indicator RAG system is now deployed on AWS with:
- ✅ Ultra-fast performance (0.6-2 seconds)
- ✅ Complete Japanese format support
- ✅ Auto-scaling capabilities
- ✅ Comprehensive monitoring
- ✅ Production-ready security

**Next Steps:**
1. Test your API endpoints
2. Monitor performance metrics
3. Set up alerts for errors
4. Scale based on usage patterns 