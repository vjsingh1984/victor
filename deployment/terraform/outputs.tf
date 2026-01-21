# Outputs for Victor AI infrastructure

output "vpc_id" {
  description = "VPC ID"
  value       = try(module.vpc.vpc_id, data.aws_vpc.selected.id)
}

output "cluster_id" {
  description = "EKS cluster ID"
  value       = try(module.eks.cluster_id, data.aws_eks_cluster.cluster.id)
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = try(module.eks.cluster_endpoint, data.aws_eks_cluster.cluster.endpoint)
}

output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = try(module.eks.cluster_security_group_id, "")
}

output "cluster_iam_role_arn" {
  description = "EKS cluster IAM role ARN"
  value       = try(module.eks.cluster_iam_role_arn, "")
}

output "node_group_iam_role_arn" {
  description = "EKS node group IAM role ARN"
  value       = try(module.eks.eks_managed_node_groups["main"].iam_role_arn, "")
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = module.rds.db_instance_endpoint
  sensitive   = true
}

output "database_port" {
  description = "RDS instance port"
  value       = module.rds.db_instance_port
}

output "database_id" {
  description = "RDS instance ID"
  value       = module.rds.db_instance_id
}

output "redis_endpoint" {
  description = "ElastiCache replication group endpoint"
  value       = module.elasticache.replication_group_id_endpoint[0]
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache port"
  value       = 6379
}

output "security_group_id" {
  description = "Security group ID"
  value       = module.security_group.security_group_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = var.create_vpc ? module.vpc.private_subnets : data.aws_subnets.private.ids
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = var.create_vpc ? module.vpc.public_subnets : []
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "helm_release_status" {
  description = "Helm release status"
  value       = helm_release.victor_ai.status
}

output "helm_release_name" {
  description = "Helm release name"
  value       = helm_release.victor_ai.name
}

output "helm_release_namespace" {
  description = "Helm release namespace"
  value       = helm_release.victor_ai.namespace
}

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = try(aws_lb.victor_ai[0].dns_name, "")
}

output "load_balancer_zone_id" {
  description = "Load balancer zone ID"
  value       = try(aws_lb.victor_ai[0].zone_id, "")
}

output "configure_kubectl" {
  description = "Configure kubectl command"
  value       = <<-EOT
    aws eks update-kubeconfig --name ${try(module.eks.cluster_name, var.cluster_name)} --region ${var.aws_region}
  EOT
}
