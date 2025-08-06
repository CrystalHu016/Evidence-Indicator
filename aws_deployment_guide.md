# AWS Deployment Guide for Evidence Indicator RAG System

## 🚀 Deployment Options

### Option 1: AWS Lambda + API Gateway (Serverless)
- **Best for**: Low to medium traffic, cost-effective
- **Pros**: Pay-per-use, auto-scaling, no server management
- **Cons**: Cold start latency, 15-minute timeout limit

### Option 2: AWS ECS/Fargate (Containerized)
- **Best for**: Medium to high traffic, consistent performance
- **Pros**: No cold starts, better for long-running processes
- **Cons**: Higher cost, requires container management

### Option 3: AWS EC2 (Traditional)
- **Best for**: High traffic, full control, custom configurations
- **Pros**: Full control, predictable performance
- **Cons**: Server management, higher operational overhead

## 📋 Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** (for containerized deployment)
4. **Python 3.9+** on local machine

## 🔧 Required AWS Services

- **Lambda** (for serverless)
- **API Gateway** (for HTTP endpoints)
- **ECS/Fargate** (for containerized)
- **EC2** (for traditional)
- **S3** (for data storage)
- **Secrets Manager** (for API keys)
- **CloudWatch** (for monitoring)

## 🛠 Quick Start: Lambda Deployment

### Step 1: Prepare Lambda Package
```bash
# Create deployment package
mkdir lambda_package
cp -r *.py lambda_package/
cp requirements.txt lambda_package/
cd lambda_package

# Install dependencies
pip install -r requirements.txt -t .

# Create ZIP file
zip -r ../evidence_indicator_lambda.zip .
```

### Step 2: Create Lambda Function
```bash
# Create Lambda function
aws lambda create-function \
  --function-name evidence-indicator-rag \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://evidence_indicator_lambda.zip \
  --timeout 300 \
  --memory-size 1024
```

### Step 3: Set Environment Variables
```bash
aws lambda update-function-configuration \
  --function-name evidence-indicator-rag \
  --environment Variables='{
    "OPENAI_API_KEY":"your-openai-key",
    "CHROMA_PATH":"/tmp/chroma",
    "DATA_PATH":"/tmp/data"
  }'
```

## 🐳 Docker Deployment (ECS/Fargate)

### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /app/chroma /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py"]
```

### Step 2: Build and Push Docker Image
```bash
# Build image
docker build -t evidence-indicator-rag .

# Tag for ECR
docker tag evidence-indicator-rag:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/evidence-indicator-rag:latest

# Push to ECR
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/evidence-indicator-rag:latest
```

## 🌐 API Gateway Setup

### Step 1: Create REST API
```bash
# Create API
aws apigateway create-rest-api \
  --name evidence-indicator-api \
  --description "Evidence Indicator RAG API"

# Get API ID
API_ID=$(aws apigateway get-rest-apis --query 'items[?name==`evidence-indicator-api`].id' --output text)

# Create resource
aws apigateway create-resource \
  --rest-api-id $API_ID \
  --parent-id $(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text) \
  --path-part "query"

# Create POST method
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/query`].id' --output text) \
  --http-method POST \
  --authorization-type NONE
```

## 📊 Monitoring and Testing

### CloudWatch Metrics
- **Invocation Count**: Number of API calls
- **Duration**: Response time
- **Error Rate**: Failed requests
- **Memory Usage**: Resource utilization

### Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test with 100 requests, 10 concurrent
ab -n 100 -c 10 -H "Content-Type: application/json" \
  -p test_payload.json \
  https://YOUR_API_ID.execute-api.REGION.amazonaws.com/prod/query
```

## 🔐 Security Best Practices

### 1. Secrets Management
```bash
# Store OpenAI API key in Secrets Manager
aws secretsmanager create-secret \
  --name evidence-indicator/openai-key \
  --secret-string '{"OPENAI_API_KEY":"your-key-here"}'
```

### 2. IAM Roles and Policies
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:evidence-indicator/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:REGION:ACCOUNT:*"
    }
  ]
}
```

### 3. VPC Configuration (if needed)
- Place Lambda/ECS in private subnets
- Use NAT Gateway for outbound internet access
- Configure security groups appropriately

## 💰 Cost Optimization

### Lambda Pricing (us-east-1)
- **Requests**: $0.20 per 1M requests
- **Compute**: $0.0000166667 per GB-second
- **Estimated cost**: ~$0.50-2.00 per 1000 queries

### ECS/Fargate Pricing
- **CPU**: $0.04048 per vCPU per hour
- **Memory**: $0.004445 per GB per hour
- **Estimated cost**: ~$30-50 per month for 24/7 operation

## 🧪 Testing Strategy

### 1. Unit Tests
```bash
# Run existing tests
python -m pytest test_all_queries.py -v
```

### 2. Integration Tests
```bash
# Test API endpoints
curl -X POST https://YOUR_API_URL/query \
  -H "Content-Type: application/json" \
  -d '{"query": "コンバインとは何ですか"}'
```

### 3. Performance Tests
```bash
# Test response times
for i in {1..10}; do
  time curl -X POST https://YOUR_API_URL/query \
    -H "Content-Type: application/json" \
    -d '{"query": "コンバインとは何ですか"}'
done
```

## 🚨 Troubleshooting

### Common Issues
1. **Cold Start Latency**: Use provisioned concurrency
2. **Memory Issues**: Increase Lambda memory allocation
3. **Timeout Errors**: Optimize code or increase timeout
4. **API Key Issues**: Check Secrets Manager configuration

### Debug Commands
```bash
# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/evidence-indicator

# Get recent log events
aws logs filter-log-events \
  --log-group-name /aws/lambda/evidence-indicator-rag \
  --start-time $(date -d '1 hour ago' +%s)000
```

## 📈 Scaling Considerations

### Auto-scaling Triggers
- **CPU Utilization**: Scale when >70%
- **Memory Usage**: Scale when >80%
- **Request Count**: Scale when >1000 requests/minute

### Performance Targets
- **Response Time**: <2 seconds (95th percentile)
- **Throughput**: 1000+ requests/minute
- **Availability**: 99.9% uptime

## 🎯 Next Steps

1. **Choose deployment option** based on your requirements
2. **Set up AWS infrastructure** using provided scripts
3. **Deploy the application** and configure monitoring
4. **Run comprehensive tests** to validate performance
5. **Monitor and optimize** based on real-world usage 