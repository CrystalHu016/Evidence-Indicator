#!/bin/bash

# AWS Lambda Deployment Script for Evidence Indicator RAG System
# Author: CrystalHu940106
# Date: 2025

set -e

echo "🚀 Evidence Indicator RAG - AWS Lambda Deployment"
echo "================================================"

# Configuration
FUNCTION_NAME="evidence-indicator-rag"
REGION="us-east-1"
RUNTIME="python3.9"
HANDLER="lambda_handler.lambda_handler"
TIMEOUT=300
MEMORY_SIZE=1024

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AWS CLI and credentials verified${NC}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}📋 AWS Account ID: ${ACCOUNT_ID}${NC}"

# Create deployment package
echo -e "${YELLOW}📦 Creating deployment package...${NC}"

# Clean up previous package
rm -rf lambda_package
rm -f evidence_indicator_lambda.zip

# Create package directory
mkdir lambda_package

# Copy Python files
cp *.py lambda_package/
cp requirements.txt lambda_package/

# Copy data files (if they exist)
if [ -d "data" ]; then
    cp -r data lambda_package/
fi

# Copy chroma directory (if it exists)
if [ -d "chroma" ]; then
    cp -r chroma lambda_package/
fi

cd lambda_package

# Install dependencies
echo -e "${YELLOW}📥 Installing dependencies...${NC}"
pip install -r requirements.txt -t .

# Add boto3 for AWS SDK
pip install boto3 -t .

# Create ZIP file
echo -e "${YELLOW}🗜️  Creating ZIP package...${NC}"
zip -r ../evidence_indicator_lambda.zip .

cd ..

echo -e "${GREEN}✅ Deployment package created: evidence_indicator_lambda.zip${NC}"

# Check if function exists
FUNCTION_EXISTS=$(aws lambda list-functions --region $REGION --query "Functions[?FunctionName=='$FUNCTION_NAME'].FunctionName" --output text)

if [ -z "$FUNCTION_EXISTS" ]; then
    echo -e "${YELLOW}🆕 Creating new Lambda function...${NC}"
    
    # Create execution role (if it doesn't exist)
    ROLE_NAME="evidence-indicator-lambda-role"
    ROLE_EXISTS=$(aws iam list-roles --query "Roles[?RoleName=='$ROLE_NAME'].RoleName" --output text)
    
    if [ -z "$ROLE_EXISTS" ]; then
        echo -e "${YELLOW}🔐 Creating IAM role...${NC}"
        
        # Create trust policy
        cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
        
        # Create role
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document file://trust-policy.json
        
        # Attach basic execution policy
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        
        # Attach Secrets Manager policy
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
        
        # Wait for role to be available
        echo -e "${YELLOW}⏳ Waiting for IAM role to be available...${NC}"
        sleep 10
        
        rm trust-policy.json
    fi
    
    # Create Lambda function
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --role arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME \
        --handler $HANDLER \
        --zip-file fileb://evidence_indicator_lambda.zip \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $REGION
    
    echo -e "${GREEN}✅ Lambda function created successfully${NC}"
    
else
    echo -e "${YELLOW}🔄 Updating existing Lambda function...${NC}"
    
    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://evidence_indicator_lambda.zip \
        --region $REGION
    
    # Update function configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $REGION
    
    echo -e "${GREEN}✅ Lambda function updated successfully${NC}"
fi

# Set environment variables
echo -e "${YELLOW}🔧 Setting environment variables...${NC}"

# Check if OpenAI API key is provided
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set. Please set it as an environment variable or in AWS Secrets Manager.${NC}"
    echo -e "${YELLOW}💡 You can set it later using:${NC}"
    echo -e "${YELLOW}   aws lambda update-function-configuration --function-name $FUNCTION_NAME --environment Variables='{\"OPENAI_API_KEY\":\"your-key-here\"}'${NC}"
else
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --environment Variables="{
            \"OPENAI_API_KEY\":\"$OPENAI_API_KEY\",
            \"CHROMA_PATH\":\"/tmp/chroma\",
            \"DATA_PATH\":\"/tmp/data\"
        }" \
        --region $REGION
    
    echo -e "${GREEN}✅ Environment variables set${NC}"
fi

# Create API Gateway (optional)
read -p "🤔 Do you want to create an API Gateway for HTTP access? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🌐 Creating API Gateway...${NC}"
    
    # Create REST API
    API_ID=$(aws apigateway create-rest-api \
        --name evidence-indicator-api \
        --description "Evidence Indicator RAG API" \
        --region $REGION \
        --query 'id' --output text)
    
    echo -e "${GREEN}✅ API Gateway created with ID: $API_ID${NC}"
    
    # Get root resource ID
    ROOT_ID=$(aws apigateway get-resources \
        --rest-api-id $API_ID \
        --region $REGION \
        --query 'items[?path==`/`].id' --output text)
    
    # Create /query resource
    QUERY_ID=$(aws apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $ROOT_ID \
        --path-part "query" \
        --region $REGION \
        --query 'id' --output text)
    
    # Create POST method
    aws apigateway put-method \
        --rest-api-id $API_ID \
        --resource-id $QUERY_ID \
        --http-method POST \
        --authorization-type NONE \
        --region $REGION
    
    # Create integration
    aws apigateway put-integration \
        --rest-api-id $API_ID \
        --resource-id $QUERY_ID \
        --http-method POST \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri arn:aws:apigateway:$REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$FUNCTION_NAME/invocations \
        --region $REGION
    
    # Add Lambda permission for API Gateway
    aws lambda add-permission \
        --function-name $FUNCTION_NAME \
        --statement-id apigateway-permission \
        --action lambda:InvokeFunction \
        --principal apigateway.amazonaws.com \
        --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT_ID:$API_ID/*/*/*" \
        --region $REGION
    
    # Deploy API
    aws apigateway create-deployment \
        --rest-api-id $API_ID \
        --stage-name prod \
        --region $REGION
    
    API_URL="https://$API_ID.execute-api.$REGION.amazonaws.com/prod/query"
    echo -e "${GREEN}✅ API Gateway deployed successfully${NC}"
    echo -e "${GREEN}🌐 API URL: $API_URL${NC}"
    
    # Test the API
    echo -e "${YELLOW}🧪 Testing the API...${NC}"
    curl -X POST $API_URL \
        -H "Content-Type: application/json" \
        -d '{"query": "コンバインとは何ですか"}' \
        --max-time 30 || echo -e "${YELLOW}⚠️  API test failed (this is normal if no data is loaded)${NC}"
fi

# Clean up
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
rm -rf lambda_package
rm -f evidence_indicator_lambda.zip

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${GREEN}📋 Function Name: $FUNCTION_NAME${NC}"
echo -e "${GREEN}🌍 Region: $REGION${NC}"
echo -e "${GREEN}⏱️  Timeout: ${TIMEOUT}s${NC}"
echo -e "${GREEN}💾 Memory: ${MEMORY_SIZE}MB${NC}"

if [ ! -z "$API_URL" ]; then
    echo -e "${GREEN}🌐 API URL: $API_URL${NC}"
    echo -e "${GREEN}📝 Test command:${NC}"
    echo -e "${YELLOW}   curl -X POST $API_URL -H \"Content-Type: application/json\" -d '{\"query\": \"コンバインとは何ですか\"}'${NC}"
fi

echo -e "${GREEN}📊 Monitor your function in the AWS Lambda console${NC}" 