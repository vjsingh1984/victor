# Victor AI Infrastructure as Code
# Terraform configuration for deploying Victor AI on AWS

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }

  backend "s3" {
    bucket         = "victor-ai-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "victor-ai-terraform-locks"
  }
}

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

# Data sources
data "aws_eks_cluster" "cluster" {
  name = var.cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = var.cluster_name
}

data "aws_vpc" "selected" {
  tags = {
    Name = "${var.project}-${var.environment}-vpc"
  }
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.selected.id]
  }

  tags = {
    Type = "private"
  }
}

# Module: VPC and Networking (optional - if creating new VPC)
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

  tags = {
    Environment = var.environment
    Project     = var.project
  }
}

# Module: EKS Cluster (optional - if creating new cluster)
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  create_eks = var.create_cluster

  cluster_name    = "${var.project}-${var.environment}-eks"
  cluster_version = var.kubernetes_version

  vpc_id     = var.create_vpc ? module.vpc.vpc_id : data.aws_vpc.selected.id
  subnet_ids = var.create_vpc ? module.vpc.private_subnets : data.aws_subnets.private.ids

  eks_managed_node_groups = {
    main = {
      name = "${var.project}-node-group"

      instance_types = var.instance_types
      capacity_type  = var.capacity_type

      min_size     = var.min_node_count
      max_size     = var.max_node_count
      desired_size = var.desired_node_count

      disk_size = var.node_disk_size

      # Enable IMDSv2
      metadata_options = {
        http_endpoint               = "enabled"
        http_tokens                 = "required"
        http_put_response_hop_limit = 2
      }

      labels = {
        Environment = var.environment
        Project     = var.project
        NodeGroup   = "main"
      }

      tags = {
        Name = "${var.project}-${var.environment}-node"
      }
    }
  }

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # IRSA for ALB
  enable_irsa = true

  tags = {
    Environment = var.environment
    Project     = var.project
  }
}

# Module: RDS PostgreSQL
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${var.project}-${var.environment}-postgres"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.db_instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_encrypted     = true
  kms_key_id           = var.kms_key_id

  db_name  = "victor"
  username = var.db_username
  port     = 5432

  vpc_security_group_ids = [module.security_group.security_group_id]
  subnet_ids             = var.create_vpc ? module.vpc.database_subnets : data.aws_subnets.private.ids

  maintenance_window = "Mon:03:00-Mon:04:00"
  backup_window      = "03:00-06:00"
  backup_retention_period = 7

  performance_insights_enabled = var.environment == "production"

  monitoring_interval = var.environment == "production" ? 60 : 0

  create_random_password = true
  random_password_length = 32

  parameters = [
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    },
    {
      name  = "log_statement"
      value = "all"
    }
  ]

  tags = {
    Environment = var.environment
    Project     = var.project
  }
}

# Module: ElastiCache Redis
module "elasticache" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  create_replication_group = true
  replication_group_id     = "${var.project}-${var.environment}-redis"
  replication_group_description = "Redis for ${var.project} ${var.environment}"

  engine                    = "redis"
  engine_version            = "7.0"
  node_type                 = var.redis_node_type
  num_cache_clusters        = var.redis_num_nodes
  automatic_failover_enabled = var.environment == "production"

  multi_az_enabled = var.environment == "production"

  subnet_ids  = var.create_vpc ? module.vpc.database_subnets : data.aws_subnets.private.ids
  vpc_id      = var.create_vpc ? module.vpc.vpc_id : data.aws_vpc.selected.id

  auth_token = var.redis_auth_token

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window          = "03:00-05:00"

  parameter_group_name = "default.redis7"

  tags = {
    Environment = var.environment
    Project     = var.project
  }
}

# Security Group
module "security_group" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"

  name        = "${var.project}-${var.environment}-sg"
  description = "Security group for Victor AI"
  vpc_id      = var.create_vpc ? module.vpc.vpc_id : data.aws_vpc.selected.id

  ingress_with_cidr_blocks = [
    {
      rule        = "postgresql-tcp"
      cidr_blocks = var.create_vpc ? module.vpc.private_subnets_cidr_blocks : data.aws_vpc.selected.cidr_block
    },
    {
      rule        = "redis-tcp"
      cidr_blocks = var.create_vpc ? module.vpc.private_subnets_cidr_blocks : data.aws_vpc.selected.cidr_block
    }
  ]

  egress_with_cidr_blocks = [
    {
      rule        = "all-all"
      cidr_blocks = "0.0.0.0/0"
    }
  ]

  tags = {
    Environment = var.environment
    Project     = var.project
  }
}

# Helm release for Victor AI
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
    name  = "replicaCount"
    value = var.replica_count
  }

  set {
    name  = "config.profile"
    value = var.environment
  }

  set {
    name  = "secrets.database.url"
    value = "postgresql://${module.rds.db_username}:${module.rds.db_password}@${module.rds.db_instance_endpoint}/${module.rds.db_instance_endpoint}"
  }

  set {
    name  = "secrets.redis.url"
    value = "redis://${module.elasticache.replication_group_id_endpoint.0}:6379"
  }

  values = [
    file("./deployment/helm/values-${var.environment}.yaml")
  ]

  depends_on = [module.rds, module.elasticache]

  lifecycle {
    ignore_changes = [
      set[0].value  # Ignore image tag changes during manual updates
    ]
  }
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = var.create_vpc ? module.vpc.vpc_id : data.aws_vpc.selected.id
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = var.create_cluster ? module.eks.cluster_endpoint : data.aws_eks_cluster.cluster.endpoint
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache replication group endpoint"
  value       = module.elasticache.replication_group_id_endpoint
  sensitive   = true
}

output "load_balancer_url" {
  description = "Load balancer URL"
  value       = "https://victor-ai-${var.environment}.example.com"
}
