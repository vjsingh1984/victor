# Terraform Deployment Guide for Victor AI

This directory contains Terraform configurations for deploying Victor AI infrastructure on AWS.

## Overview

The Terraform configuration provisions:
- **VPC** (optional) - Virtual Private Cloud with public/private subnets
- **EKS Cluster** (optional) - Kubernetes cluster for container orchestration
- **RDS PostgreSQL** - Managed PostgreSQL database
- **ElastiCache Redis** - Managed Redis cache
- **Security Groups** - Network security rules
- **Helm Release** - Victor AI deployment via Helm

## Prerequisites

1. **Terraform** 1.5+ installed
2. **AWS CLI** configured with credentials
3. **kubectl** configured for EKS access
4. **Helm** 3.x installed
5. Existing VPC and EKS cluster (if not creating new ones)

## Quick Start

### 1. Initialize Terraform

```bash
cd deployment/terraform

# Initialize backend and providers
terraform init

# Select workspace
terraform workspace new production
# or
terraform workspace select production
```

### 2. Configure Variables

Create `terraform.tfvars`:

```hcl
environment = "production"
aws_region  = "us-east-1"

# Use existing infrastructure
create_vpc     = false
create_cluster = false
cluster_name   = "victor-ai-prod-eks"

# Or create new infrastructure
# create_vpc     = true
# create_cluster = true
# vpc_cidr       = "10.0.0.0/16"

# Image configuration
image_tag     = "0.5.0"
replica_count = 6

# Database configuration
db_instance_class  = "db.t3.micro"
allocated_storage  = 20
max_allocated_storage = 1000

# Redis configuration
redis_node_type = "cache.t3.micro"
redis_num_nodes = 2
```

### 3. Plan and Apply

```bash
# Review changes
terraform plan

# Apply changes
terraform apply

# Auto-approve (use with caution)
terraform apply -auto-approve
```

### 4. Configure kubectl

```bash
# Update kubeconfig
terraform output configure_kubectl | bash

# Or manually
aws eks update-kubeconfig --name victor-ai-production-eks --region us-east-1
```

### 5. Verify Deployment

```bash
# Check nodes
kubectl get nodes

# Check pods
kubectl get pods -n victor-ai-production

# Check services
kubectl get svc -n victor-ai-production
```

## Configuration

### Workspaces

Use Terraform workspaces for multiple environments:

```bash
# Development
terraform workspace new development
terraform apply -var="environment=development"

# Staging
terraform workspace new staging
terraform apply -var="environment=staging"

# Production
terraform workspace new production
terraform apply -var="environment=production"
```

### Provider Configuration

```hcl
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "victor-ai"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", data.aws_eks_cluster.cluster.name]
  }
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", data.aws_eks_cluster.cluster.name]
    }
  }
}
```

### Backend Configuration

```hcl
terraform {
  backend "s3" {
    bucket         = "victor-ai-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "victor-ai-terraform-locks"
  }
}
```

Create S3 bucket and DynamoDB table:

```bash
# Create S3 bucket
aws s3api create-bucket \
  --bucket victor-ai-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket victor-ai-terraform-state \
  --versioning-configuration Status=Enabled

# Create DynamoDB table
aws dynamodb create-table \
  --table-name victor-ai-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region us-east-1
```

## Modules

### VPC Module

Create a new VPC with public/private subnets:

```hcl
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  create_vpc = var.create_vpc

  name = "${var.project}-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "development"
  enable_dns_hostnames = true
  enable_dns_support   = true
}
```

### EKS Module

Create a new EKS cluster:

```hcl
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project}-${var.environment}-eks"
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    main = {
      name = "${var.project}-node-group"

      instance_types = var.instance_types
      capacity_type  = var.capacity_type

      min_size     = var.min_node_count
      max_size     = var.max_node_count
      desired_size = var.desired_node_count
    }
  }
}
```

### RDS Module

Create managed PostgreSQL database:

```hcl
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${var.project}-${var.environment}-postgres"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_encrypted     = true

  db_name  = "victor"
  username = var.db_username

  vpc_security_group_ids = [module.security_group.security_group_id]
  subnet_ids             = module.vpc.database_subnets
}
```

### ElastiCache Module

Create managed Redis cache:

```hcl
module "elasticache" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  replication_group_id     = "${var.project}-${var.environment}-redis"
  engine                   = "redis"
  engine_version           = "7.0"
  node_type                = var.redis_node_type
  num_cache_clusters       = var.redis_num_nodes
  automatic_failover_enabled = var.environment == "production"

  subnet_ids  = module.vpc.database_subnets
  vpc_id      = module.vpc.vpc_id
}
```

### Helm Release

Deploy Victor AI via Helm:

```hcl
resource "helm_release" "victor_ai" {
  name       = "victor-ai"
  repository = "./deployment/helm"
  chart      = "victor-ai"
  namespace  = "victor-ai-${var.environment}"

  create_namespace = true

  set {
    name  = "image.tag"
    value = var.image_tag
  }

  set {
    name  = "secrets.database.url"
    value = module.rds.db_instance_endpoint
  }

  set {
    name  = "secrets.redis.url"
    value = module.elasticache.replication_group_id_endpoint[0]
  }
}
```

## Operations

### Planning

```bash
# Basic plan
terraform plan

# Plan with specific variables
terraform plan -var="environment=production" -var="replica_count=10"

# Plan with target (specific resource only)
terraform plan -target=module.rds

# Plan with out file (for review)
terraform plan -out=tfplan

# Review saved plan
terraform show tfplan
```

### Applying

```bash
# Apply saved plan
terraform apply tfplan

# Apply with auto-approve
terraform apply -auto-approve

# Apply with specific variables
terraform apply -var="environment=production"

# Apply with target (use with caution)
terraform apply -target=module.rds
```

### Destroying

```bash
# Plan destroy
terraform plan -destroy

# Destroy all resources
terraform destroy

# Destroy with auto-approve
terraform destroy -auto-approve

# Destroy specific resources
terraform destroy -target=module.rds
```

### State Management

```bash
# View state
terraform show

# List resources in state
terraform state list

# Show specific resource
terraform state show 'module.rds.aws_db_instance.this[0]'

# Remove resource from state (without destroying)
terraform state rm 'module.rds.aws_db_instance.this[0]'

# Move resource in state
terraform state mv 'module.old.module.rds' 'module.new.module.rds'

# Import existing resource
terraform import 'module.rds.aws_db_instance.this[0]' victor-ai-production-postgres
```

### Outputs

```bash
# View all outputs
terraform output

# View specific output
terraform output cluster_endpoint

# View output in JSON format
terraform output -json cluster_endpoint

# Use output in shell
export CLUSTER_ENDPOINT=$(terraform output cluster_endpoint)
```

## Multi-Region Deployment

### Configure Provider Aliases

```hcl
provider "aws" {
  region = "us-east-1"
  alias  = "us_east_1"
}

provider "aws" {
  region = "eu-west-1"
  alias  = "eu_west_1"
}
```

### Deploy to Multiple Regions

```bash
# Deploy to us-east-1
terraform workspace new us-east-1
terraform apply -var="aws_region=us-east-1"

# Deploy to eu-west-1
terraform workspace new eu-west-1
terraform apply -var="aws_region=eu-west-1"
```

## Cost Optimization

### Use Spot Instances

```hcl
module "eks" {
  eks_managed_node_groups = {
    spot = {
      name = "spot-node-group"
      capacity_type = "SPOT"
      instance_types = ["t3a.large", "t3.large", "m5.large"]
    }
  }
}
```

### Auto-Scaling

```hcl
module "eks" {
  eks_managed_node_groups = {
    main = {
      min_size     = 3
      max_size     = 50
      desired_size = 6
    }
  }
}
```

### Right-Sizing Resources

```hcl
# Development
db_instance_class = "db.t3.micro"
redis_node_type   = "cache.t3.micro"

# Production
db_instance_class = "db.r6g.xlarge"
redis_node_type   = "cache.r6g.large"
```

## Security

### Encryption

```hcl
# RDS encryption
module "rds" {
  storage_encrypted = true
  kms_key_id        = var.kms_key_id
}

# ElastiCache encryption
module "elasticache" {
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token
}
```

### Network Security

```hcl
module "security_group" {
  source  = "terraform-aws-modules/security-group/aws"

  ingress_with_cidr_blocks = [
    {
      rule        = "postgresql-tcp"
      cidr_blocks = module.vpc.private_subnets_cidr_blocks
    }
  ]
}
```

### IAM Roles

```hcl
module "eks" {
  # IRSA for pod IAM roles
  enable_irsa = true

  # Create IAM roles for service accounts
  iam_role_arns = {
    victor-ai = module.victor_ai_role.iam_role_arn
  }
}
```

## Monitoring and Logging

### Enable CloudWatch

```hcl
module "eks" {
  cluster_enabled_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]
}
```

### Enable Enhanced Monitoring

```hcl
module "rds" {
  monitoring_interval = 60
  monitoring_role_arn = var.monitoring_role_arn
}
```

### Performance Insights

```hcl
module "rds" {
  performance_insights_enabled = var.environment == "production"
}
```

## Backup and Recovery

### Automated Backups

```hcl
module "rds" {
  backup_retention_period = 7
  backup_window          = "03:00-06:00"
  maintenance_window     = "Mon:03:00-Mon:04:00"
}
```

### Snapshot Management

```hcl
resource "aws_db_snapshot" "victor_ai_snapshot" {
  db_instance_identifier = module.rds.db_instance_id
  db_snapshot_identifier = "victor-ai-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
}

resource "aws_backup_vault" "victor_ai" {
  name = "victor-ai-backup-vault"
}

resource "aws_backup_plan" "victor_ai" {
  name = "victor-ai-backup-plan"

  rule {
    rule_name           = "victor-ai-backup-rule"
    target_vault_name   = aws_backup_vault.victor_ai.name
    schedule_expression = "cron(0 3 * * ? *)"

    lifecycle {
      delete_after = 30
    }
  }
}
```

## Troubleshooting

### State Lock Issues

```bash
# Force unlock (use with caution!)
terraform force-unlock <LOCK_ID>

# Check DynamoDB locks
aws dynamodb scan --table-name victor-ai-terraform-locks
```

### Resource Creation Failures

```bash
# Check CloudTrail logs
aws cloudtrail lookup-events --lookup-attributes AttributeKey=ResourceName,AttributeValue=victor-ai

# Check CloudWatch logs
aws logs tail /aws/eks/victor-ai-production/cluster --follow
```

### Provider Issues

```bash
# Reinitialize providers
terraform init -upgrade

# Check provider versions
terraform providers

# Validate configuration
terraform validate
```

## Best Practices

1. **Use workspaces** for environment separation
2. **Enable state locking** with DynamoDB
3. **Use remote backend** (S3) for state storage
4. **Enable versioning** on state bucket
5. **Use resource targeting** carefully
6. **Plan before apply** in production
7. **Tag all resources** for cost allocation
8. **Use modules** for reusability
9. **Enable encryption** everywhere
10. **Implement least privilege** IAM policies

## Additional Resources

- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [EKS Best Practices Guide](https://aws.github.io/aws-eks-best-practices/)
- [RDS User Guide](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/)
- [ElastiCache Documentation](https://docs.aws.amazon.com/AmazonElastiCache/latest/)
- [Helm Provider](https://registry.terraform.io/providers/hashicorp/helm/latest/docs)
