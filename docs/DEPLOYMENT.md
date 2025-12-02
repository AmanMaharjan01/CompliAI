# Deployment Guide

## Local Development

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Manual Setup

```bash
# Start database
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Initialize database
python scripts/init_db.py

# Start API
uvicorn src.api.main:app --reload
```

## Production Deployment (AWS)

### Prerequisites

- AWS Account
- AWS CLI configured
- Docker installed
- Terraform (optional)

### Step 1: Build and Push Docker Image

```bash
# Build image
docker build -t policy-assistant:latest .

# Tag for ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag policy-assistant:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/policy-assistant:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/policy-assistant:latest
```

### Step 2: Deploy to ECS

Create ECS cluster, task definition, and service using AWS Console or Terraform.

### Step 3: Configure RDS PostgreSQL

```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier policy-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username admin \
    --master-user-password <password> \
    --allocated-storage 20
```

### Step 4: Set Environment Variables

Configure in ECS Task Definition or use AWS Secrets Manager:

- OPENAI_API_KEY
- LANGCHAIN_API_KEY
- COHERE_API_KEY
- DATABASE_URL
- JWT_SECRET_KEY

### Step 5: Configure Load Balancer

Create Application Load Balancer and target group pointing to ECS service.

## Monitoring

- CloudWatch Logs for application logs
- CloudWatch Metrics for performance
- LangSmith for LLM tracing
- Prometheus + Grafana for custom metrics

## Scaling

### Horizontal Scaling

```yaml
# ECS Service auto-scaling
TargetTrackingScalingPolicy:
  TargetValue: 70 # CPU utilization
  ScaleInCooldown: 300
  ScaleOutCooldown: 60
```

### Database Scaling

- Enable Multi-AZ for RDS
- Use Read Replicas for heavy read workloads
- Consider Aurora Serverless for auto-scaling

## Backup and Recovery

```bash
# Database backup
aws rds create-db-snapshot \
    --db-instance-identifier policy-db \
    --db-snapshot-identifier policy-db-backup-$(date +%Y%m%d)

# Vector store backup
tar -czf vector_store_backup.tar.gz data/vector_store/
aws s3 cp vector_store_backup.tar.gz s3://backups/
```

## Security Best Practices

1. Use AWS Secrets Manager for sensitive data
2. Enable VPC for private networking
3. Use IAM roles instead of access keys
4. Enable encryption at rest and in transit
5. Regular security audits
6. Implement WAF for API protection
