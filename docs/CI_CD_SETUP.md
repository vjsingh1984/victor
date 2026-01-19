# CI/CD Setup Guide for Victor AI

Complete guide for setting up CI/CD pipelines for Victor AI.

## Table of Contents

- [Overview](#overview)
- [GitHub Actions](#github-actions)
- [GitLab CI](#gitlab-ci)
- [Jenkins](#jenkins)
- [Best Practices](#best-practices)

## Overview

Victor AI includes pre-configured CI/CD pipelines for:

- Automated testing
- Docker image building
- Kubernetes deployment
- Helm chart deployment
- Multi-environment deployments

## GitHub Actions

### Available Workflows

Located in `.github/workflows/`:

1. **ci.yml** - Continuous Integration
2. **test.yml** - Testing
3. **deploy.yml** - Package deployment
4. **deploy-k8s.yml** - Kubernetes deployment
5. **lint.yml** - Code quality checks
6. **security.yml** - Security scanning
7. **release.yml** - Release automation

### Setting Up GitHub Actions

#### Required Secrets

Add these secrets to your GitHub repository settings:

```
DOCKER_USERNAME          # Docker Hub username
DOCKER_PASSWORD          # Docker Hub password
AWS_ACCESS_KEY_ID        # AWS access key (for EKS)
AWS_SECRET_ACCESS_KEY    # AWS secret key (for EKS)
KUBE_CONFIG              # Base64-encoded kubeconfig file
ANTHROPIC_API_KEY        # Anthropic API key (for tests)
OPENAI_API_KEY           # OpenAI API key (for tests)
```

#### Manual Workflow Trigger

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        type: choice
        options:
          - staging
          - production
      cluster:
        description: 'Kubernetes cluster'
        required: true
        type: choice
        options:
          - staging-cluster
          - production-cluster
```

Trigger from GitHub Actions UI:
- Go to Actions tab
- Select "Deploy to Kubernetes" workflow
- Click "Run workflow"
- Select environment and cluster

### Customizing Workflows

#### Modify Deploy Workflow

Edit `.github/workflows/deploy-k8s.yml`:

```yaml
- name: Deploy via Helm
  run: |
    helm upgrade --install victor ./config/helm/victor \
      --set image.tag="${{ steps.version.outputs.version }}" \
      --set replicaCount=5 \
      --namespace victor
```

#### Add Custom Steps

```yaml
- name: Run custom tests
  run: |
    python -m pytest tests/integration/ -v

- name: Notify Slack
  if: success()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## GitLab CI

### Creating .gitlab-ci.yml

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  IMAGE_NAME: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
  HELM_CHART: ./config/helm/victor

test:
  stage: test
  image: python:3.12
  script:
    - pip install -e ".[dev]"
    - pytest tests/unit/ -v
    - pytest tests/integration/ -v

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $IMAGE_NAME .
    - docker push $IMAGE_NAME

deploy:staging:
  stage: deploy
  image: alpine/helm:3.13
  environment:
    name: staging
    url: https://staging.victor.example.com
  script:
    - helm upgrade --install victor $HELM_CHART \
        --set image.tag=$CI_COMMIT_SHORT_SHA \
        --namespace victor-staging \
        --create-namespace
  only:
    - main

deploy:production:
  stage: deploy
  image: alpine/helm:3.13
  environment:
    name: production
    url: https://victor.example.com
  script:
    - helm upgrade --install victor $HELM_CHART \
        --set image.tag=$CI_COMMIT_SHORT_SHA \
        --namespace victor \
        --create-namespace
  when: manual
  only:
    - tags
```

### GitLab CI Variables

Set these in GitLab project settings (Settings > CI/CD > Variables):

```
CI_REGISTRY_USER          # Container registry username
CI_REGISTRY_PASSWORD      # Container registry password
KUBE_CONFIG               # Base64-encoded kubeconfig
HELM_REPO_USERNAME        # Helm repo username (if private)
HELM_REPO_PASSWORD        # Helm repo password (if private)
```

## Jenkins

### Creating Jenkinsfile

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        IMAGE_NAME = "vijayksingh/victor:${BUILD_NUMBER}"
        KUBECONFIG = credentials('kubeconfig')
        DOCKER_CREDENTIALS = credentials('docker-hub')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            agent {
                docker {
                    image 'python:3.12'
                }
            }
            steps {
                sh 'pip install -e ".[dev]"'
                sh 'pytest tests/unit/ -v'
                sh 'pytest tests/integration/ -v'
            }
        }

        stage('Build') {
            steps {
                script {
                    docker.withRegistry('', 'docker-hub') {
                        docker.build('vijayksingh/victor:${BUILD_NUMBER}')
                        docker.image('vijayksingh/victor:${BUILD_NUMBER}').push()
                    }
                }
            }
        }

        stage('Deploy Staging') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    helm upgrade --install victor ./config/helm/victor \
                        --set image.tag=${BUILD_NUMBER} \
                        --namespace victor-staging \
                        --create-namespace
                """
            }
        }

        stage('Deploy Production') {
            when {
                tag pattern: "v\\d+\\.\\d+\\.\\d+", comparator: "REGEXP"
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                sh """
                    helm upgrade --install victor ./config/helm/victor \
                        --set image.tag=${BUILD_NUMBER} \
                        --namespace victor
                """
            }
        }
    }

    post {
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
```

### Jenkins Configuration

#### Required Plugins

- Docker Pipeline Plugin
- Kubernetes CLI Plugin
- Helm Pipeline Plugin

#### Credentials

Add these credentials in Jenkins (Manage Jenkins > Manage Credentials):

- `docker-hub`: Docker Hub credentials
- `kubeconfig`: Kubernetes configuration
- `api-keys`: LLM provider API keys

## Best Practices

### Version Tagging

```bash
# Tag releases
git tag -a v0.5.0 -m "Release v0.5.0"
git push origin v0.5.0

# Automated version bump in CI
- name: Bump version
  run: |
        VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml'))['project']['version'])")
        echo "VERSION=$VERSION" >> $GITHUB_ENV
```

### Multi-Environment Deployment

```yaml
# Staging
deploy:staging:
  environment: staging
  only:
    - main

# Production
deploy:production:
  environment: production
  when: manual
  only:
    - /^v\d+\.\d+\.\d+$/
```

### Rollback Strategy

```yaml
- name: Deploy
  run: helm upgrade --install victor ./config/helm/victor

- name: Rollback on failure
  if: failure()
  run: helm rollback victor -n victor
```

### Security Scanning

```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: vijayksingh/victor:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'

- name: Upload Trivy results to GitHub Security tab
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

### Notifications

```yaml
# Slack notification
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
  if: always()

# Email notification
- name: Send email
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: "Deployment ${{ job.status }}"
    to: team@example.com
    from: ci@example.com
```

### Caching

```yaml
# Cache Python packages
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

# Cache Docker layers
- name: Cache Docker layers
  uses: actions/cache@v3
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ github.sha }}
    restore-keys: |
      ${{ runner.os }}-buildx-
```

## Troubleshooting

### Pipeline Failures

```bash
# Check logs locally
act -l                    # List workflows
act -j test               # Run test job locally

# Debug CI
- name: Debug
  run: |
        echo "Environment:"
        env
        echo "Working directory:"
        pwd
        ls -la
```

### Deployment Issues

```bash
# Verify Helm chart
helm template victor ./config/helm/victor --debug

# Dry-run deployment
helm upgrade --install victor ./config/helm/victor \
  --dry-run --debug

# Check deployment
kubectl get all -n victor
kubectl describe deployment victor-api -n victor
```

### Rollback Procedures

```bash
# Manual rollback
helm rollback victor -n victor

# Rollback to specific revision
helm rollback victor 2 -n victor

# Kubernetes rollback
kubectl rollout undo deployment/victor-api -n victor
```

## Support

For CI/CD issues:
- GitHub Issues: https://github.com/vijayksingh/victor/issues
- Documentation: https://github.com/vijayksingh/victor#readme
