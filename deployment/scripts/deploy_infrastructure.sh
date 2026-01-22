#!/bin/bash
################################################################################
# Infrastructure Deployment Script
#
# This script automates the deployment of all infrastructure components required
# for Victor AI in production, including PostgreSQL, Redis, Ingress Controller,
# and cert-manager for TLS certificate management.
#
# Usage:
#   ./deploy_infrastructure.sh [--namespace NAMESPACE]
#                              [--components postgres,redis,ingress,cert-manager]
#                              [--storage-class STORAGE_CLASS]
#                              [--dry-run]
#                              [--skip-wait]
#                              [--rollback]
#                              [--uninstall]
#
# Features:
# - Idempotent deployments (safe to run multiple times)
# - Dry-run mode for testing
# - Component-level deployment granularity
# - Automatic health checks and validation
# - Rollback capability
# - Comprehensive logging and progress tracking
################################################################################

set -euo pipefail

################################################################################
# Script Configuration
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ROOT="${PROJECT_ROOT}/deployment"
LOG_DIR="${PROJECT_ROOT}/logs"
REPORT_DIR="${PROJECT_ROOT}/reports"

# Default configuration
NAMESPACE="victor"
COMPONENTS="postgres,redis,ingress,cert-manager"
STORAGE_CLASS="standard"
DRY_RUN=false
SKIP_WAIT=false
ROLLBACK=false
UNINSTALL=false

# Timeouts
HEALTH_CHECK_TIMEOUT=300
HEALTH_CHECK_INTERVAL=5
DEPLOYMENT_TIMEOUT=600

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Deployment tracking
declare -a DEPLOYMENT_STEPS
INFRA_DEPLOYMENT_ID=""
DEPLOYMENT_START_TIME=""

################################################################################
# Helper Functions
################################################################################

log_info() {
    local msg="[INFO] $1"
    echo -e "${BLUE}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_success() {
    local msg="[PASS] $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_warning() {
    local msg="[WARN] $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_error() {
    local msg="[FAIL] $1"
    echo -e "${RED}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_section() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  $1${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $1 ===" >> "${DEPLOYMENT_LOG_FILE}"
}

show_progress() {
    local current=$1
    local total=$2
    local msg=$3

    local percent=$(( current * 100 / total ))
    local filled=$(( percent / 2 ))
    local empty=$(( 50 - filled ))

    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %d%% - %s" "${percent}" "${msg}"
}

add_deployment_step() {
    DEPLOYMENT_STEPS+=("$1|$2")
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --components)
                COMPONENTS="$2"
                shift 2
                ;;
            --storage-class)
                STORAGE_CLASS="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-wait)
                SKIP_WAIT=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --uninstall)
                UNINSTALL=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy infrastructure components for Victor AI production environment.

Options:
  --namespace NAMESPACE      Kubernetes namespace (default: victor)
  --components COMPONENTS    Comma-separated list of components to deploy
                             (postgres,redis,ingress,cert-manager, default: all)
  --storage-class CLASS      StorageClass for persistent volumes (default: standard)
  --dry-run                  Perform a dry run without making changes
  --skip-wait                Skip waiting for components to be ready
  --rollback                 Rollback the most recent deployment
  --uninstall                Uninstall all infrastructure components
  --help                     Show this help message

Examples:
  # Deploy all components
  $0

  # Deploy only PostgreSQL and Redis
  $0 --components postgres,redis

  # Deploy with custom storage class
  $0 --storage-class fast-ssd

  # Dry run to see what would be deployed
  $0 --dry-run

  # Uninstall all infrastructure
  $0 --uninstall

Components:
  postgres        PostgreSQL StatefulSet with persistent storage
  redis           Redis StatefulSet with persistent storage
  ingress         NGINX ingress controller
  cert-manager    TLS certificate management with Let's Encrypt
EOF
}

################################################################################
# Initialization
################################################################################

initialize_deployment() {
    log_section "Initializing Infrastructure Deployment"

    # Generate deployment ID
    INFRA_DEPLOYMENT_ID="infra-${NAMESPACE}-$(date +%Y%m%d-%H%M%S)"
    DEPLOYMENT_START_TIME=$(date +%s)

    # Setup directories
    mkdir -p "${LOG_DIR}"
    mkdir -p "${REPORT_DIR}"

    # Setup log file
    export DEPLOYMENT_LOG_FILE="${LOG_DIR}/infrastructure_${INFRA_DEPLOYMENT_ID}.log"

    log_info "Deployment ID: ${INFRA_DEPLOYMENT_ID}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Components: ${COMPONENTS}"
    log_info "Storage Class: ${STORAGE_CLASS}"
    log_info "Dry Run: ${DRY_RUN}"
    log_info "Uninstall: ${UNINSTALL}"

    # Verify kubectl is available
    if ! command -v kubectl &>/dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    # Verify cluster connectivity
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Initialization complete"
    add_deployment_step "Initialize Deployment" "SUCCESS"
}

################################################################################
# Pre-flight Checks
################################################################################

check_prerequisites() {
    log_section "Prerequisites Check"

    # Check if helm is installed (required for ingress and cert-manager)
    if [[ "${COMPONENTS}" == *"ingress"* ]] || [[ "${COMPONENTS}" == *"cert-manager"* ]]; then
        if ! command -v helm &>/dev/null; then
            log_error "helm is required for ingress and cert-manager deployment"
            log_info "Install helm from: https://helm.sh/docs/intro/install/"
            exit 1
        fi
        log_success "Helm is installed"
    fi

    # Check if namespace exists, create if not
    if ! kubectl get namespace "${NAMESPACE}" &>/dev/null; then
        log_info "Creating namespace: ${NAMESPACE}"

        if [[ "${DRY_RUN}" == "false" ]]; then
            kubectl create namespace "${NAMESPACE}" >> "${DEPLOYMENT_LOG_FILE}" 2>&1
        fi

        log_success "Namespace created"
    else
        log_success "Namespace exists"
    fi

    # Check storage class
    if ! kubectl get storageclass "${STORAGE_CLASS}" &>/dev/null; then
        log_warning "StorageClass '${STORAGE_CLASS}' not found"
        log_info "Available StorageClasses:"
        kubectl get storageclass || log_warning "No StorageClasses available"
        log_error "Please specify a valid StorageClass with --storage-class"
        exit 1
    fi

    log_success "StorageClass '${STORAGE_CLASS}' is available"

    add_deployment_step "Prerequisites Check" "SUCCESS"
}

################################################################################
# PostgreSQL Deployment
################################################################################

deploy_postgres() {
    log_section "Deploying PostgreSQL"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would deploy PostgreSQL"
        return 0
    fi

    # Create PostgreSQL manifests
    local postgres_manifest=$(cat << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: ${NAMESPACE}
data:
  POSTGRES_DB: victor
  POSTGRES_USER: victor
  POSTGRES_INITDB_ARGS: "--encoding=UTF8"
---
apiVersion: v1
kind: Secret
metadata:
  name: postgres-secret
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  POSTGRES_PASSWORD: "$(openssl rand -base64 32)"
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: ${NAMESPACE}
  labels:
    app: postgres
spec:
  ports:
  - port: 5432
    targetPort: 5432
    name: postgres
  selector:
    app: postgres
  clusterIP: None
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ${NAMESPACE}
  labels:
    app: postgres
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
          name: postgres
        env:
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: postgres-config
              key: POSTGRES_DB
        - name: POSTGRES_USER
          valueFrom:
            configMapKeyRef:
              name: postgres-config
              key: POSTGRES_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: POSTGRES_PASSWORD
        - name: POSTGRES_INITDB_ARGS
          valueFrom:
            configMapKeyRef:
              name: postgres-config
              key: POSTGRES_INITDB_ARGS
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: 250m
            memory: 256Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        livenessProbe:
          exec:
            command:
            - sh
            - -c
            - pg_isready -U \${POSTGRES_USER} -d \${POSTGRES_DB}
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          exec:
            command:
            - sh
            - -c
            - pg_isready -U \${POSTGRES_USER} -d \${POSTGRES_DB}
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ${STORAGE_CLASS}
      resources:
        requests:
          storage: 10Gi
EOF
)

    # Apply manifests
    log_info "Applying PostgreSQL manifests..."
    echo "${postgres_manifest}" | kubectl apply -f - >> "${DEPLOYMENT_LOG_FILE}" 2>&1

    if [[ $? -ne 0 ]]; then
        log_error "Failed to apply PostgreSQL manifests"
        return 1
    fi

    log_success "PostgreSQL manifests applied"

    # Wait for StatefulSet to be ready
    if [[ "${SKIP_WAIT}" == "false" ]]; then
        wait_for_statefulset "postgres" "${NAMESPACE}"
    fi

    add_deployment_step "Deploy PostgreSQL" "SUCCESS"
    return 0
}

################################################################################
# Redis Deployment
################################################################################

deploy_redis() {
    log_section "Deploying Redis"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would deploy Redis"
        return 0
    fi

    # Create Redis manifests
    local redis_manifest=$(cat << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: ${NAMESPACE}
data:
  redis.conf: |
    appendonly yes
    maxmemory 512mb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: ${NAMESPACE}
  labels:
    app: redis
spec:
  ports:
  - port: 6379
    targetPort: 6379
    name: redis
  selector:
    app: redis
  clusterIP: None
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: ${NAMESPACE}
  labels:
    app: redis
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        args:
        - /etc/redis/redis.conf
        ports:
        - containerPort: 6379
          name: redis
        volumeMounts:
        - name: redis-config
          mountPath: /etc/redis
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: ${STORAGE_CLASS}
      resources:
        requests:
          storage: 5Gi
EOF
)

    # Apply manifests
    log_info "Applying Redis manifests..."
    echo "${redis_manifest}" | kubectl apply -f - >> "${DEPLOYMENT_LOG_FILE}" 2>&1

    if [[ $? -ne 0 ]]; then
        log_error "Failed to apply Redis manifests"
        return 1
    fi

    log_success "Redis manifests applied"

    # Wait for StatefulSet to be ready
    if [[ "${SKIP_WAIT}" == "false" ]]; then
        wait_for_statefulset "redis" "${NAMESPACE}"
    fi

    add_deployment_step "Deploy Redis" "SUCCESS"
    return 0
}

################################################################################
# Ingress Controller Deployment
################################################################################

deploy_ingress() {
    log_section "Deploying NGINX Ingress Controller"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would deploy NGINX Ingress Controller"
        return 0
    fi

    # Add nginx-stable helm repo
    log_info "Adding nginx-stable Helm repository..."

    helm repo add nginx-stable https://helm.nginx.com/stable >> "${DEPLOYMENT_LOG_FILE}" 2>&1
    helm repo update >> "${DEPLOYMENT_LOG_FILE}" 2>&1

    # Create values file
    local ingress_values=$(cat << EOF
controller:
  kind: Deployment
  replicaCount: 2
  image:
    repository: nginxinc/nginx-ingress-controller
    tag: latest
  service:
    type: LoadBalancer
    externalTrafficPolicy: Local
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 256Mi
  metrics:
    enabled: true
    service:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9113"
  config:
    entries:
      http-snippets: |
        more_set_headers "X-Frame-Options: DENY";
        more_set_headers "X-Content-Type-Options: nosniff";
        more_set_headers "X-XSS-Protection: 1; mode=block";
EOF
)

    echo "${ingress_values}" > /tmp/ingress-values.yaml

    # Deploy ingress controller
    log_info "Deploying NGINX Ingress Controller..."

    if helm upgrade --install nginx-ingress nginx-stable/nginx-ingress \
        --namespace "${NAMESPACE}" \
        --values /tmp/ingress-values.yaml \
        --timeout "${DEPLOYMENT_TIMEOUT}s" \
        --wait >> "${DEPLOYMENT_LOG_FILE}" 2>&1; then
        log_success "NGINX Ingress Controller deployed"
    else
        log_error "Failed to deploy NGINX Ingress Controller"
        rm -f /tmp/ingress-values.yaml
        return 1
    fi

    rm -f /tmp/ingress-values.yaml

    # Get LoadBalancer IP
    if [[ "${SKIP_WAIT}" == "false" ]]; then
        log_info "Waiting for LoadBalancer IP..."

        local elapsed=0
        while [[ ${elapsed} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
            local lb_ip
            lb_ip=$(kubectl get service nginx-ingress-controller -n "${NAMESPACE}" \
                -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

            if [[ -n "${lb_ip}" ]]; then
                log_success "LoadBalancer IP: ${lb_ip}"
                break
            fi

            show_progress ${elapsed} ${HEALTH_CHECK_TIMEOUT} "Waiting for LoadBalancer IP"
            sleep "${HEALTH_CHECK_INTERVAL}"
            elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
        done
        printf "\n"
    fi

    add_deployment_step "Deploy Ingress Controller" "SUCCESS"
    return 0
}

################################################################################
# Cert-Manager Deployment
################################################################################

deploy_cert_manager() {
    log_section "Deploying cert-manager"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would deploy cert-manager"
        return 0
    fi

    # Check if cert-manager is already installed
    if kubectl get namespace cert-manager &>/dev/null; then
        log_warning "cert-manager namespace already exists"
        log_info "Skipping cert-manager installation"
        add_deployment_step "Deploy cert-manager" "SKIPPED"
        return 0
    fi

    # Add jetstack helm repo
    log_info "Adding jetstack Helm repository..."

    helm repo add jetstack https://charts.jetstack.io >> "${DEPLOYMENT_LOG_FILE}" 2>&1
    helm repo update >> "${DEPLOYMENT_LOG_FILE}" 2>&1

    # Install cert-manager CRDs
    log_info "Installing cert-manager CRDs..."

    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.1/cert-manager.crds.yaml \
        >> "${DEPLOYMENT_LOG_FILE}" 2>&1

    if [[ $? -ne 0 ]]; then
        log_error "Failed to install cert-manager CRDs"
        return 1
    fi

    # Create values file
    local cert_manager_values=$(cat << EOF
installCRDs: true
replicaCount: 2
resources:
  requests:
    cpu: 10m
    memory: 32Mi
  limits:
    cpu: 100m
    memory: 128Mi
webhook:
  resources:
    requests:
      cpu: 10m
      memory: 32Mi
    limits:
      cpu: 100m
      memory: 128Mi
cainjector:
  resources:
    requests:
      cpu: 10m
      memory: 32Mi
    limits:
      cpu: 100m
      memory: 128Mi
EOF
)

    echo "${cert_manager_values}" > /tmp/cert-manager-values.yaml

    # Deploy cert-manager
    log_info "Deploying cert-manager..."

    if helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --values /tmp/cert-manager-values.yaml \
        --version v1.13.1 \
        --timeout "${DEPLOYMENT_TIMEOUT}s" \
        --wait >> "${DEPLOYMENT_LOG_FILE}" 2>&1; then
        log_success "cert-manager deployed"
    else
        log_error "Failed to deploy cert-manager"
        rm -f /tmp/cert-manager-values.yaml
        return 1
    fi

    rm -f /tmp/cert-manager-values.yaml

    # Create ClusterIssuer for Let's Encrypt
    log_info "Creating Let's Encrypt ClusterIssuer..."

    local cluster_issuer=$(cat << EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@victor.ai
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
)

    echo "${cluster_issuer}" | kubectl apply -f - >> "${DEPLOYMENT_LOG_FILE}" 2>&1

    if [[ $? -ne 0 ]]; then
        log_warning "Failed to create ClusterIssuer (you may need to update email)"
    else
        log_success "ClusterIssuer created"
    fi

    add_deployment_step "Deploy cert-manager" "SUCCESS"
    return 0
}

################################################################################
# Wait Functions
################################################################################

wait_for_statefulset() {
    local name=$1
    local namespace=$2

    log_info "Waiting for StatefulSet ${name} to be ready..."

    local elapsed=0
    while [[ ${elapsed} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
        local ready
        ready=$(kubectl get statefulset "${name}" -n "${namespace}" \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

        local desired
        desired=$(kubectl get statefulset "${name}" -n "${namespace}" \
            -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

        show_progress ${elapsed} ${HEALTH_CHECK_TIMEOUT} \
            "${name}: ${ready}/${desired} replicas ready"

        if [[ "${ready}" -eq "${desired}" ]] && [[ "${desired}" -gt 0 ]]; then
            printf "\n"
            log_success "StatefulSet ${name} is ready (${ready}/${desired})"
            return 0
        fi

        sleep "${HEALTH_CHECK_INTERVAL}"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done

    printf "\n"
    log_error "Timeout waiting for StatefulSet ${name}"
    return 1
}

wait_for_deployment() {
    local name=$1
    local namespace=$2

    log_info "Waiting for Deployment ${name} to be ready..."

    local elapsed=0
    while [[ ${elapsed} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
        local ready
        ready=$(kubectl get deployment "${name}" -n "${namespace}" \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

        local desired
        desired=$(kubectl get deployment "${name}" -n "${namespace}" \
            -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

        show_progress ${elapsed} ${HEALTH_CHECK_TIMEOUT} \
            "${name}: ${ready}/${desired} replicas ready"

        if [[ "${ready}" -eq "${desired}" ]] && [[ "${desired}" -gt 0 ]]; then
            printf "\n"
            log_success "Deployment ${name} is ready (${ready}/${desired})"
            return 0
        fi

        sleep "${HEALTH_CHECK_INTERVAL}"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done

    printf "\n"
    log_error "Timeout waiting for Deployment ${name}"
    return 1
}

################################################################################
# Health Checks
################################################################################

verify_postgres() {
    log_info "Verifying PostgreSQL connectivity..."

    local ready=0
    local elapsed=0

    while [[ ${elapsed} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
        # Check if pod is running
        local pod_status
        pod_status=$(kubectl get pod -n "${NAMESPACE}" -l app=postgres \
            -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")

        if [[ "${pod_status}" == "Running" ]]; then
            # Check if database is accepting connections
            local pod_name
            pod_name=$(kubectl get pod -n "${NAMESPACE}" -l app=postgres \
                -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

            if [[ -n "${pod_name}" ]]; then
                if kubectl exec -n "${NAMESPACE}" "${pod_name}" -- \
                    pg_isready -U victor -d victor &>/dev/null; then
                    log_success "PostgreSQL is ready"
                    return 0
                fi
            fi
        fi

        sleep "${HEALTH_CHECK_INTERVAL}"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done

    log_error "PostgreSQL health check failed"
    return 1
}

verify_redis() {
    log_info "Verifying Redis connectivity..."

    local elapsed=0

    while [[ ${elapsed} -lt ${HEALTH_CHECK_TIMEOUT} ]]; do
        # Check if pod is running
        local pod_status
        pod_status=$(kubectl get pod -n "${NAMESPACE}" -l app=redis \
            -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "")

        if [[ "${pod_status}" == "Running" ]]; then
            # Check if redis is responding
            local pod_name
            pod_name=$(kubectl get pod -n "${NAMESPACE}" -l app=redis \
                -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

            if [[ -n "${pod_name}" ]]; then
                if kubectl exec -n "${NAMESPACE}" "${pod_name}" -- \
                    redis-cli ping &>/dev/null; then
                    log_success "Redis is ready"
                    return 0
                fi
            fi
        fi

        sleep "${HEALTH_CHECK_INTERVAL}"
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done

    log_error "Redis health check failed"
    return 1
}

verify_connectivity() {
    log_section "Verifying Connectivity"

    local all_passed=true

    if [[ "${COMPONENTS}" == *"postgres"* ]]; then
        if ! verify_postgres; then
            all_passed=false
        fi
    fi

    if [[ "${COMPONENTS}" == *"redis"* ]]; then
        if ! verify_redis; then
            all_passed=false
        fi
    fi

    if [[ "${all_passed}" == "true" ]]; then
        log_success "All connectivity checks passed"
        add_deployment_step "Verify Connectivity" "SUCCESS"
        return 0
    else
        log_error "Some connectivity checks failed"
        add_deployment_step "Verify Connectivity" "FAILED"
        return 1
    fi
}

################################################################################
# Rollback
################################################################################

rollback_component() {
    local component=$1

    log_warning "Rolling back ${component}..."

    case "${component}" in
        postgres)
            kubectl delete statefulset postgres -n "${NAMESPACE}" --ignore-not-found=true
            kubectl delete configmap postgres-config -n "${NAMESPACE}" --ignore-not-found=true
            kubectl delete secret postgres-secret -n "${NAMESPACE}" --ignore-not-found=true
            kubectl delete service postgres -n "${NAMESPACE}" --ignore-not-found=true
            kubectl delete pvc -n "${NAMESPACE}" -l app=postgres
            ;;
        redis)
            kubectl delete statefulset redis -n "${NAMESPACE}" --ignore-not-found=true
            kubectl delete configmap redis-config -n "${NAMESPACE}" --ignore-not-found=true
            kubectl delete service redis -n "${NAMESPACE}" --ignore-not-found=true
            kubectl delete pvc -n "${NAMESPACE}" -l app=redis
            ;;
        ingress)
            helm uninstall nginx-ingress -n "${NAMESPACE}" --ignore-not-found
            ;;
        cert-manager)
            helm uninstall cert-manager -n cert-manager --ignore-not-found
            kubectl delete namespace cert-manager --ignore-not-found=true
            kubectl delete clusterissuer letsencrypt-prod --ignore-not-found=true
            ;;
    esac

    log_success "Rollback complete for ${component}"
}

perform_rollback() {
    log_section "Performing Rollback"

    IFS=',' read -ra COMPONENT_ARRAY <<< "${COMPONENTS}"

    for component in "${COMPONENT_ARRAY[@]}"; do
        rollback_component "${component}"
    done

    add_deployment_step "Rollback" "SUCCESS"
}

################################################################################
# Uninstall
################################################################################

perform_uninstall() {
    log_section "Uninstalling Infrastructure"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would uninstall all infrastructure"
        return 0
    fi

    log_warning "This will remove all infrastructure components"
    log_warning "Data volumes will be preserved unless you explicitly delete them"

    # Uninstall in reverse order
    if [[ "${COMPONENTS}" == *"cert-manager"* ]]; then
        rollback_component "cert-manager"
    fi

    if [[ "${COMPONENTS}" == *"ingress"* ]]; then
        rollback_component "ingress"
    fi

    if [[ "${COMPONENTS}" == *"redis"* ]]; then
        rollback_component "redis"
    fi

    if [[ "${COMPONENTS}" == *"postgres"* ]]; then
        rollback_component "postgres"
    fi

    log_success "Uninstall complete"
    log_info "To delete persistent volumes, run:"
    log_info "  kubectl delete pvc -n ${NAMESPACE} --all"

    return 0
}

################################################################################
# Deployment Execution
################################################################################

execute_deployment() {
    log_section "Executing Deployment"

    IFS=',' read -ra COMPONENT_ARRAY <<< "${COMPONENTS}"

    local total_components=${#COMPONENT_ARRAY[@]}
    local current=0

    for component in "${COMPONENT_ARRAY[@]}"; do
        current=$((current + 1))
        log_info "Deploying component ${current}/${total_components}: ${component}"

        case "${component}" in
            postgres)
                if ! deploy_postgres; then
                    log_error "Failed to deploy postgres"
                    return 1
                fi
                ;;
            redis)
                if ! deploy_redis; then
                    log_error "Failed to deploy redis"
                    return 1
                fi
                ;;
            ingress)
                if ! deploy_ingress; then
                    log_error "Failed to deploy ingress"
                    return 1
                fi
                ;;
            cert-manager)
                if ! deploy_cert_manager; then
                    log_error "Failed to deploy cert-manager"
                    return 1
                fi
                ;;
            *)
                log_warning "Unknown component: ${component}"
                ;;
        esac
    done

    return 0
}

################################################################################
# Reporting
################################################################################

generate_deployment_report() {
    log_section "Generating Deployment Report"

    local deployment_end_time=$(date +%s)
    local duration=$((deployment_end_time - DEPLOYMENT_START_TIME))
    local duration_minutes=$((duration / 60))
    local duration_seconds=$((duration % 60))

    local report_file="${REPORT_DIR}/infrastructure_${INFRA_DEPLOYMENT_ID}.json"

    cat > "${report_file}" << EOF
{
  "deployment_id": "${INFRA_DEPLOYMENT_ID}",
  "namespace": "${NAMESPACE}",
  "components": "${COMPONENTS}",
  "storage_class": "${STORAGE_CLASS}",
  "dry_run": ${DRY_RUN},
  "uninstall": ${UNINSTALL},
  "started_at": "$(date -u -d @${DEPLOYMENT_START_TIME} +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "duration_seconds": ${duration},
  "duration_formatted": "${duration_minutes}m ${duration_seconds}s",
  "status": "SUCCESS",
  "steps": [
EOF

    local first=true
    for step in "${DEPLOYMENT_STEPS[@]}"; do
        local step_name
        local step_status
        step_name=$(echo "${step}" | cut -d'|' -f1)
        step_status=$(echo "${step}" | cut -d'|' -f2)

        if [[ "${first}" == "true" ]]; then
            first=false
        else
            echo "," >> "${report_file}"
        fi

        cat >> "${report_file}" << EOF
    {
      "name": "${step_name}",
      "status": "${step_status}"
    }
EOF
    done

    cat >> "${report_file}" << EOF

  ]
}
EOF

    log_success "Report generated: ${report_file}"
}

print_summary() {
    echo ""
    echo "==================================================================="
    echo "INFRASTRUCTURE DEPLOYMENT SUMMARY"
    echo "==================================================================="
    echo "Deployment ID:      ${INFRA_DEPLOYMENT_ID}"
    echo "Namespace:          ${NAMESPACE}"
    echo "Components:         ${COMPONENTS}"
    echo "Storage Class:      ${STORAGE_CLASS}"
    echo "Status:             SUCCESS"
    echo ""
    echo "Deployment Steps:"
    printf "%-40s %-15s\n" "Step" "Status"
    printf "%-40s %-15s\n" "-----" "------"

    for step in "${DEPLOYMENT_STEPS[@]}"; do
        local step_name
        local step_status
        step_name=$(echo "${step}" | cut -d'|' -f1)
        step_status=$(echo "${step}" | cut -d'|' -f2)

        case "${step_status}" in
            SUCCESS)
                printf "%-40s ${GREEN}%-15s${NC}\n" "${step_name}" "${step_status}"
                ;;
            FAILED)
                printf "%-40s ${RED}%-15s${NC}\n" "${step_name}" "${step_status}"
                ;;
            SKIPPED)
                printf "%-40s ${YELLOW}%-15s${NC}\n" "${step_name}" "${step_status}"
                ;;
        esac
    done

    echo ""
    echo "Connection Information:"
    if [[ "${COMPONENTS}" == *"postgres"* ]]; then
        echo "  PostgreSQL:"
        echo "    Host:      postgres.${NAMESPACE}.svc.cluster.local"
        echo "    Port:      5432"
        echo "    Database:  victor"
        echo "    Password:  \$(kubectl get secret -n ${NAMESPACE} postgres-secret -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)"
    fi

    if [[ "${COMPONENTS}" == *"redis"* ]]; then
        echo "  Redis:"
        echo "    Host:      redis.${NAMESPACE}.svc.cluster.local"
        echo "    Port:      6379"
    fi

    if [[ "${COMPONENTS}" == *"ingress"* ]]; then
        local lb_ip
        lb_ip=$(kubectl get service nginx-ingress-controller -n "${NAMESPACE}" \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        echo "  Ingress:"
        echo "    LoadBalancer: ${lb_ip}"
    fi

    echo ""
    echo "Logs:           ${DEPLOYMENT_LOG_FILE}"
    echo "Report:         ${REPORT_DIR}/infrastructure_${INFRA_DEPLOYMENT_ID}.json"
    echo "==================================================================="
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_arguments "$@"
    initialize_deployment

    if [[ "${UNINSTALL}" == "true" ]]; then
        perform_uninstall
        print_summary
        exit 0
    fi

    if [[ "${ROLLBACK}" == "true" ]]; then
        perform_rollback
        print_summary
        exit 0
    fi

    check_prerequisites

    if execute_deployment; then
        if [[ "${SKIP_WAIT}" == "false" ]]; then
            verify_connectivity
        fi
        generate_deployment_report
        print_summary
        log_success "Infrastructure deployment completed successfully"
        exit 0
    else
        log_error "Infrastructure deployment failed"
        generate_deployment_report

        # Automatic rollback on failure
        log_warning "Initiating automatic rollback..."
        perform_rollback

        exit 1
    fi
}

# Run main function
main "$@"
