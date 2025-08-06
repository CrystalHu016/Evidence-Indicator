#!/bin/bash

# AWS ECS/Fargate Deployment Script for Evidence Indicator RAG System
# Author: CrystalHu940106
# Date: 2025

set -e

echo "🐳 Evidence Indicator RAG - AWS ECS/Fargate Deployment"
echo "====================================================="

# Configuration
CLUSTER_NAME="evidence-indicator-cluster"
SERVICE_NAME="evidence-indicator-service"
TASK_DEFINITION_NAME="evidence-indicator-task"
REGION="us-east-1"
ECR_REPOSITORY_NAME="evidence-indicator-rag"
IMAGE_TAG="latest"

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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites verified${NC}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}📋 AWS Account ID: ${ACCOUNT_ID}${NC}"

# Create ECR repository
echo -e "${YELLOW}🏗️  Creating ECR repository...${NC}"
ECR_REPOSITORY_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPOSITORY_NAME"

# Check if repository exists
REPO_EXISTS=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $REGION 2>/dev/null || echo "NOT_FOUND")

if [ "$REPO_EXISTS" = "NOT_FOUND" ]; then
    aws ecr create-repository \
        --repository-name $ECR_REPOSITORY_NAME \
        --region $REGION
    
    echo -e "${GREEN}✅ ECR repository created: $ECR_REPOSITORY_URI${NC}"
else
    echo -e "${GREEN}✅ ECR repository already exists: $ECR_REPOSITORY_URI${NC}"
fi

# Login to ECR
echo -e "${YELLOW}🔐 Logging in to ECR...${NC}"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# Build Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker build -t $ECR_REPOSITORY_NAME:$IMAGE_TAG .

# Tag image for ECR
echo -e "${YELLOW}🏷️  Tagging image for ECR...${NC}"
docker tag $ECR_REPOSITORY_NAME:$IMAGE_TAG $ECR_REPOSITORY_URI:$IMAGE_TAG

# Push image to ECR
echo -e "${YELLOW}📤 Pushing image to ECR...${NC}"
docker push $ECR_REPOSITORY_URI:$IMAGE_TAG

echo -e "${GREEN}✅ Docker image pushed successfully${NC}"

# Create ECS cluster
echo -e "${YELLOW}🏗️  Creating ECS cluster...${NC}"
CLUSTER_EXISTS=$(aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION --query 'clusters[0].status' --output text 2>/dev/null || echo "INACTIVE")

if [ "$CLUSTER_EXISTS" = "INACTIVE" ] || [ "$CLUSTER_EXISTS" = "None" ]; then
    aws ecs create-cluster \
        --cluster-name $CLUSTER_NAME \
        --region $REGION
    
    echo -e "${GREEN}✅ ECS cluster created: $CLUSTER_NAME${NC}"
else
    echo -e "${GREEN}✅ ECS cluster already exists: $CLUSTER_NAME${NC}"
fi

# Create task execution role
echo -e "${YELLOW}🔐 Creating task execution role...${NC}"
ROLE_NAME="evidence-indicator-ecs-task-execution-role"
ROLE_EXISTS=$(aws iam list-roles --query "Roles[?RoleName=='$ROLE_NAME'].RoleName" --output text)

if [ -z "$ROLE_EXISTS" ]; then
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
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
    
    # Attach task execution policy
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    
    # Wait for role to be available
    echo -e "${YELLOW}⏳ Waiting for IAM role to be available...${NC}"
    sleep 10
    
    rm trust-policy.json
    echo -e "${GREEN}✅ Task execution role created${NC}"
else
    echo -e "${GREEN}✅ Task execution role already exists${NC}"
fi

# Create task definition
echo -e "${YELLOW}📋 Creating task definition...${NC}"

# Check if OpenAI API key is provided
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set. Please set it as an environment variable.${NC}"
    echo -e "${YELLOW}💡 You can set it using: export OPENAI_API_KEY='your-key-here'${NC}"
    ENVIRONMENT_VARS="[]"
else
    ENVIRONMENT_VARS="[
        {
            \"name\": \"OPENAI_API_KEY\",
            \"value\": \"$OPENAI_API_KEY\"
        },
        {
            \"name\": \"CHROMA_PATH\",
            \"value\": \"/app/chroma\"
        },
        {
            \"name\": \"DATA_PATH\",
            \"value\": \"/app/data\"
        },
        {
            \"name\": \"PORT\",
            \"value\": \"8000\"
        }
    ]"
fi

# Create task definition JSON
cat > task-definition.json << EOF
{
  "family": "$TASK_DEFINITION_NAME",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME",
  "containerDefinitions": [
    {
      "name": "evidence-indicator-container",
      "image": "$ECR_REPOSITORY_URI:$IMAGE_TAG",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": $ENVIRONMENT_VARS,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/$TASK_DEFINITION_NAME",
          "awslogs-region": "$REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

# Register task definition
aws ecs register-task-definition \
    --cli-input-json file://task-definition.json \
    --region $REGION

echo -e "${GREEN}✅ Task definition registered${NC}"

# Create CloudWatch log group
echo -e "${YELLOW}📊 Creating CloudWatch log group...${NC}"
aws logs create-log-group \
    --log-group-name "/ecs/$TASK_DEFINITION_NAME" \
    --region $REGION 2>/dev/null || echo -e "${GREEN}✅ Log group already exists${NC}"

# Create Application Load Balancer (optional)
read -p "🤔 Do you want to create an Application Load Balancer for HTTP access? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🌐 Creating Application Load Balancer...${NC}"
    
    # Create VPC (if needed)
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)
    echo -e "${GREEN}✅ Using VPC: $VPC_ID${NC}"
    
    # Get subnets
    SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0:2].SubnetId' --output text --region $REGION)
    SUBNET_1=$(echo $SUBNET_IDS | cut -d' ' -f1)
    SUBNET_2=$(echo $SUBNET_IDS | cut -d' ' -f2)
    
    # Create security group
    SG_NAME="evidence-indicator-sg"
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group for Evidence Indicator RAG" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' --output text)
    
    # Add inbound rules
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    echo -e "${GREEN}✅ Security group created: $SG_ID${NC}"
    
    # Create ALB
    ALB_NAME="evidence-indicator-alb"
    ALB_ARN=$(aws elbv2 create-load-balancer \
        --name $ALB_NAME \
        --subnets $SUBNET_1 $SUBNET_2 \
        --security-groups $SG_ID \
        --region $REGION \
        --query 'LoadBalancers[0].LoadBalancerArn' --output text)
    
    echo -e "${GREEN}✅ Application Load Balancer created: $ALB_ARN${NC}"
    
    # Get ALB DNS name
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --load-balancer-arns $ALB_ARN \
        --region $REGION \
        --query 'LoadBalancers[0].DNSName' --output text)
    
    # Create target group
    TG_NAME="evidence-indicator-tg"
    TG_ARN=$(aws elbv2 create-target-group \
        --name $TG_NAME \
        --protocol HTTP \
        --port 8000 \
        --vpc-id $VPC_ID \
        --target-type ip \
        --health-check-path /health \
        --region $REGION \
        --query 'TargetGroups[0].TargetGroupArn' --output text)
    
    # Create listener
    aws elbv2 create-listener \
        --load-balancer-arn $ALB_ARN \
        --protocol HTTP \
        --port 80 \
        --default-actions Type=forward,TargetGroupArn=$TG_ARN \
        --region $REGION
    
    echo -e "${GREEN}✅ Target group and listener created${NC}"
    
    # Create ECS service with ALB
    echo -e "${YELLOW}🚀 Creating ECS service with ALB...${NC}"
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_DEFINITION_NAME \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
        --load-balancers "targetGroupArn=$TG_ARN,containerName=evidence-indicator-container,containerPort=8000" \
        --region $REGION
    
    echo -e "${GREEN}✅ ECS service created with ALB${NC}"
    echo -e "${GREEN}🌐 ALB URL: http://$ALB_DNS${NC}"
    
else
    # Create ECS service without ALB
    echo -e "${YELLOW}🚀 Creating ECS service without ALB...${NC}"
    
    # Get default VPC and subnets
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)
    SUBNET_IDS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0:2].SubnetId' --output text --region $REGION)
    SUBNET_1=$(echo $SUBNET_IDS | cut -d' ' -f1)
    SUBNET_2=$(echo $SUBNET_IDS | cut -d' ' -f2)
    
    # Create security group
    SG_NAME="evidence-indicator-sg"
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group for Evidence Indicator RAG" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' --output text)
    
    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name $SERVICE_NAME \
        --task-definition $TASK_DEFINITION_NAME \
        --desired-count 1 \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_1,$SUBNET_2],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
        --region $REGION
    
    echo -e "${GREEN}✅ ECS service created (no ALB)${NC}"
    echo -e "${YELLOW}💡 You can access the service using ECS task public IP${NC}"
fi

# Clean up
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
rm -f task-definition.json

echo -e "${GREEN}🎉 ECS/Fargate deployment completed successfully!${NC}"
echo -e "${GREEN}📋 Cluster Name: $CLUSTER_NAME${NC}"
echo -e "${GREEN}🚀 Service Name: $SERVICE_NAME${NC}"
echo -e "${GREEN}📦 Task Definition: $TASK_DEFINITION_NAME${NC}"
echo -e "${GREEN}🐳 Container Image: $ECR_REPOSITORY_URI:$IMAGE_TAG${NC}"
echo -e "${GREEN}🌍 Region: $REGION${NC}"

if [ ! -z "$ALB_DNS" ]; then
    echo -e "${GREEN}🌐 ALB URL: http://$ALB_DNS${NC}"
    echo -e "${GREEN}📝 Test command:${NC}"
    echo -e "${YELLOW}   curl -X POST http://$ALB_DNS/query -H \"Content-Type: application/json\" -d '{\"query\": \"コンバインとは何ですか\"}'${NC}"
fi

echo -e "${GREEN}📊 Monitor your service in the AWS ECS console${NC}"
echo -e "${GREEN}🔍 View logs in CloudWatch: /ecs/$TASK_DEFINITION_NAME${NC}" 