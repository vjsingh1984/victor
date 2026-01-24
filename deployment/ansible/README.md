# Ansible Deployment Guide for Victor AI

This directory contains Ansible playbooks for deploying Victor AI on Linux servers.

## Overview

The Ansible deployment automates:
- **System preparation** - Package installation, user creation
- **Docker setup** - Docker engine installation and configuration
- **Victor AI deployment** - Container deployment and configuration
- **Database setup** - PostgreSQL installation and configuration
- **Redis setup** - Redis installation and configuration
- **Nginx proxy** - Reverse proxy configuration
- **Monitoring stack** - Prometheus and Grafana setup

## Prerequisites

1. **Ansible** 2.12+ installed on control machine
2. **Python** 3.8+ on target hosts
3. **SSH access** to target hosts
4. **Sudo privileges** on target hosts

## Quick Start

### 1. Install Ansible

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install ansible

# On macOS
brew install ansible

# Using pip
pip install ansible
```

### 2. Configure Inventory

Edit `deployment/ansible/inventory.ini`:

```ini
[production]
victor-ai-prod-1 ansible_host=10.0.1.10 ansible_user=ubuntu
victor-ai-prod-2 ansible_host=10.0.1.11 ansible_user=ubuntu
victor-ai-prod-3 ansible_host=10.0.1.12 ansible_user=ubuntu

[production:vars]
victor_profile=production
victor_workers=8
db_host=victor-ai-db-prod
redis_host=victor-ai-redis-prod
```

### 3. Configure Variables

Create `group_vars/all.yml`:

```yaml
victor_version: "0.5.0"
victor_user: "victor"
docker_image: "victorai/victor:0.5.0"

# Database
db_host: "localhost"
db_port: 5432
db_name: "victor"
db_user: "victor"
db_password: "{{ vault_db_password }}"

# API Keys
anthropic_api_key: "{{ vault_anthropic_api_key }}"
openai_api_key: "{{ vault_openai_api_key }}"
```

### 4. Deploy

```bash
# Check connectivity
ansible -i deployment/ansible/inventory.ini all -m ping

# Run playbook
ansible-playbook -i deployment/ansible/inventory.ini deployment/ansible/playbook.yml

# Deploy to specific environment
ansible-playbook -i deployment/ansible/inventory.ini deployment/ansible/playbook.yml --limit production
```

## Configuration

### Inventory Structure

```
deployment/ansible/
├── inventory.ini              # Static inventory file
├── playbook.yml               # Main playbook
├── group_vars/
│   ├── all.yml                # Variables for all hosts
│   ├── production.yml         # Production-specific variables
│   ├── staging.yml            # Staging-specific variables
│   └── development.yml        # Development-specific variables
├── host_vars/
│   ├── victor-ai-prod-1.yml   # Host-specific variables
│   └── victor-ai-prod-2.yml
└── templates/                 # Jinja2 templates
    ├── environment.j2
    ├── victor_config.yml.j2
    ├── victor-ai.service.j2
    └── nginx.conf.j2
```

### Dynamic Inventory

Use AWS EC2 dynamic inventory:

```bash
# Install boto3
pip install boto3

# Create AWS inventory file
cat > inventory.aws.yml <<EOF
plugin: aws_ec2
regions:
  - us-east-1
filters:
  tag:Environment: production
  tag:Application: victor-ai
 keyed_groups:
  - key: tags.Environment
    prefix: env_
  - key: tags.Application
    prefix: app_
EOF

# Test dynamic inventory
ansible-inventory -i inventory.aws.yml --list
```

### Variables

#### System Variables

```yaml
# User configuration
victor_user: "victor"
victor_group: "victor"
victor_install_dir: "/opt/victor-ai"
victor_config_dir: "/etc/victor-ai"
victor_data_dir: "/var/lib/victor-ai"
victor_log_dir: "/var/log/victor-ai"

# Docker configuration
docker_registry: "docker.io"
docker_image: "victorai/victor"
docker_tag: "{{ victor_version }}"
```

#### Service Variables

```yaml
# Service configuration
victor_port: 8000
victor_workers: 4
victor_log_level: "INFO"
victor_profile: "production"

# Resource limits
victor_memory_limit: "2g"
victor_memory_reservation: "512m"
victor_cpu_limit: "2.0"
victor_cpu_reservation: "0.5"
```

#### Database Variables

```yaml
# PostgreSQL
db_host: "localhost"
db_port: 5432
db_name: "victor"
db_user: "victor"
db_password: "{{ vault_db_password }}"

# Connection string
database_url: "postgresql://{{ db_user }}:{{ db_password }}@{{ db_host }}:{{ db_port }}/{{ db_name }}"
```

#### Redis Variables

```yaml
# Redis
redis_host: "localhost"
redis_port: 6379
redis_password: "{{ vault_redis_password }}"

# Connection string
redis_url: "redis://:{{ redis_password }}@{{ redis_host }}:{{ redis_port }}/0"
```

### Secrets Management

#### Using Ansible Vault

```bash
# Create encrypted file
ansible-vault create group_vars/all/vault.yml

# Edit encrypted file
ansible-vault edit group_vars/all/vault.yml

# View encrypted file
ansible-vault view group_vars/all/vault.yml

# Change password
ansible-vault rekey group_vars/all/vault.yml
```

`group_vars/all/vault.yml`:

```yaml
vault_db_password: "secretpassword"
vault_redis_password: "secretpassword"
vault_anthropic_api_key: "sk-ant-..."
vault_openai_api_key: "sk-..."
```

Run playbook with vault:

```bash
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass
```

Or use vault password file:

```bash
echo "myvaultpassword" > .vault_pass
chmod 600 .vault_pass

ansible-playbook -i inventory.ini playbook.yml --vault-password-file .vault_pass
```

#### Using External Secrets

```yaml
# Pull secrets from HashiCorp Vault
- name: Get secrets from Vault
  ansible.builtin.uri:
    url: "https://vault.example.com/v1/secret/data/victor-ai"
    method: GET
    headers:
      X-Vault-Token: "{{ vault_token }}"
    return_content: yes
  register: vault_secrets
  no_log: true

- name: Set secrets
  set_fact:
    db_password: "{{ vault_secrets.json.data.data.db_password }}"
    api_key: "{{ vault_secrets.json.data.data.api_key }}"
```

## Playbook Execution

### Basic Execution

```bash
# Run entire playbook
ansible-playbook -i inventory.ini playbook.yml

# Run specific tags
ansible-playbook -i inventory.ini playbook.yml --tags docker

# Skip specific tags
ansible-playbook -i inventory.ini playbook.yml --skip-tags monitoring

# Run with extra variables
ansible-playbook -i inventory.ini playbook.yml -e "victor_version=0.5.2"

# Run with specific hosts
ansible-playbook -i inventory.ini playbook.yml --limit victor-ai-prod-1
```

### Check Mode

```bash
# Dry-run (check mode)
ansible-playbook -i inventory.ini playbook.yml --check

# Check mode with diff
ansible-playbook -i inventory.ini playbook.yml --check --diff
```

### Verbosity

```bash
# Verbose output
ansible-playbook -i inventory.ini playbook.yml -v

# More verbose
ansible-playbook -i inventory.ini playbook.yml -vvv

# Most verbose (debugging)
ansible-playbook -i inventory.ini playbook.yml -vvvv
```

### Parallel Execution

```bash
# Run with 10 forks
ansible-playbook -i inventory.ini playbook.yml -f 10

# Run serially (one host at a time)
ansible-playbook -i inventory.ini playbook.yml --serial 1
```

## Rolling Updates

### Update Victor AI

```bash
# Pull new image and restart
ansible-playbook -i inventory.ini playbook.yml \
  -e "victor_version=0.5.2" \
  --tags update \
  --limit production

# Rolling update (one host at a time)
ansible-playbook -i inventory.ini playbook.yml \
  -e "victor_version=0.5.2" \
  --tags update \
  --limit production \
  --serial 1
```

### Rollback

```bash
# Rollback to previous version
ansible-playbook -i inventory.ini playbook.yml \
  -e "victor_version=0.5.0" \
  --tags update \
  --limit production
```

## Monitoring and Maintenance

### Check Service Status

```bash
# Check if service is running
ansible -i inventory.ini production -m shell -a "systemctl status victor-ai"

# Check service logs
ansible -i inventory.ini production -m shell -a "journalctl -u victor-ai -n 100"

# Check Docker containers
ansible -i inventory.ini production -m shell -a "docker ps"
```

### Restart Services

```bash
# Restart Victor AI
ansible -i inventory.ini production -m service -a "name=victor-ai state=restarted"

# Restart Nginx
ansible -i inventory.ini production -m service -a "name=nginx state=restarted"
```

### Health Checks

```bash
# Run health check
ansible -i inventory.ini production -m shell -a "curl -f http://localhost:8000/health"

# Run custom health check script
ansible -i inventory.ini production -m script -a "scripts/health_check.sh"
```

## Scaling

### Add New Hosts

```bash
# Add new hosts to inventory.ini
echo "victor-ai-prod-4 ansible_host=10.0.1.13 ansible_user=ubuntu" >> inventory.ini

# Deploy to new hosts
ansible-playbook -i inventory.ini playbook.yml --limit victor-ai-prod-4
```

### Remove Hosts

```bash
# Stop service on hosts to be removed
ansible -i inventory.ini production --limit victor-ai-prod-4 -m service -a "name=victor-ai state=stopped"

# Remove from load balancer
ansible -i inventory.ini production --limit victor-ai-prod-4 -m shell -a "lb_remove"
```

## Backup and Recovery

### Backup Configuration

```bash
# Backup configuration files
ansible -i inventory.ini production -m synchronize \
  -a "src=/etc/victor-ai dest=/backups/victor-ai-config mode=pull"

# Backup database
ansible -i inventory.ini victor_ai_db_servers -m shell \
  -a "pg_dump victor > /backups/victor-$(date +%F).sql"
```

### Restore Configuration

```bash
# Restore configuration files
ansible -i inventory.ini production -m synchronize \
  -a "src=/backups/victor-ai-config dest=/etc/victor-ai mode=push"

# Restore database
ansible -i inventory.ini victor_ai_db_servers -m shell \
  -a "psql victor < /backups/victor-2024-01-20.sql"
```

## Troubleshooting

### Connection Issues

```bash
# Test SSH connection
ansible -i inventory.ini all -m ping

# Test with verbose output
ansible -i inventory.ini all -m ping -vvv

# Check SSH key
ansible -i inventory.ini all -m ping --private-key ~/.ssh/custom_key
```

### Failed Playbook Runs

```bash
# Re-run from failed task
ansible-playbook -i inventory.ini playbook.yml --start-at-task "Configure Nginx"

# Ignore errors (use with caution)
ansible-playbook -i inventory.ini playbook.yml -e "ansible_skip_tags: problematic"

# Continue on error
ansible-playbook -i inventory.ini playbook.yml -e "ansible_ignore_errors=yes"
```

### Debugging

```bash
# Enable debug task
- name: Debug variables
  debug:
    var: ansible_facts

# Debug specific variable
- name: Debug variable
  debug:
    var: victor_version

# Check if condition matches
- name: Check condition
  debug:
    msg: "Condition matched"
  when: victor_version == "0.5.0"
```

## Best Practices

1. **Use version control** for playbooks and inventory
2. **Encrypt secrets** with Ansible Vault
3. **Test in staging** before production
4. **Use check mode** for dry-runs
5. **Implement idempotency** for all tasks
6. **Use handlers** for service restarts
7. **Tag tasks** for granular execution
8. **Limit scope** with --limit
9. **Use roles** for modularity
10. **Document variables** thoroughly

## Additional Resources

- [Ansible Documentation](https://docs.ansible.com/)
- [Ansible Best Practices](https://docs.ansible.com/ansible/latest/user_guide/playbooks_best_practices.html)
- [Docker Modules](https://docs.ansible.com/ansible/latest/collections/community/docker/docker_container_module.html)
- [PostgreSQL Modules](https://docs.ansible.com/ansible/latest/collections/community/postgresql/postgresql_db_module.html)
