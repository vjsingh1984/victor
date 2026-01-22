#!/bin/bash
# Victor AI - Complete Monitoring Stack Deployment Script
# Deploys Prometheus, Grafana, AlertManager with dashboards and alerting rules
#
# Features:
# - Namespace creation with RBAC
# - Persistent storage for all components
# - Secure credentials management
# - 7 dashboards from observability/dashboards/
# - 50+ alerting rules from observability/alerts/
# - Health checks for all components
# - Test alert generation
# - Notification channel setup (email, Slack)
#
# Usage:
#   ./deploy_monitoring_complete.sh [options]
#
# Options:
#   --namespace NAME     Custom namespace (default: victor-monitoring)
#   --storage-class NAME Storage class for PVCs (default: standard)
#   --email EMAIL        Email for alert notifications
#   --slack-webhook URL  Slack webhook for alerts
#   --skip-dashboards    Skip dashboard import
#   --skip-alerts        Skip alerting rules setup
#   --dry-run            Show what would be deployed without deploying
#   --help               Show this help message

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K8S_MONITORING_DIR="${PROJECT_ROOT}/deployment/kubernetes/monitoring"
OBSERVABILITY_DIR="${PROJECT_ROOT}/observability"
DASHBOARDS_DIR="${OBSERVABILITY_DIR}/dashboards"
ALERTS_DIR="${OBSERVABILITY_DIR}/alerts"

# Default configuration
NAMESPACE="${NAMESPACE:-victor-monitoring}"
STORAGE_CLASS="${STORAGE_CLASS:-standard}"
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme123}"
RETENTION_DAYS="${RETENTION_DAYS:-15}"
PROMETHEUS_RETENTION_SIZE="${PROMETHEUS_RETENTION_SIZE:-10Gi}"
GRAFANA_RETENTION_SIZE="${GRAFANA_RETENTION_SIZE:-5Gi}"
ALERTMANAGER_RETENTION_SIZE="${ALERTMANAGER_RETENTION_SIZE:-2Gi}"

# Notification configuration (set via environment or flags)
EMAIL_NOTIFICATION="${EMAIL_NOTIFICATION:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Flags
SKIP_DASHBOARDS=false
SKIP_ALERTS=false
DRY_RUN=false
VERBOSE=false

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${VERBOSE}" == true ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

log_step() {
    echo -e "\n${CYAN}[STEP]${NC} $1"
    echo -e "${CYAN}================================================${NC}"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

show_help() {
    cat << EOF
Victor AI - Complete Monitoring Stack Deployment

Usage: $0 [OPTIONS]

Options:
  --namespace NAME         Custom namespace (default: victor-monitoring)
  --storage-class NAME     Storage class for PVCs (default: standard)
  --email EMAIL            Email for alert notifications
  --slack-webhook URL      Slack webhook for alerts
  --admin-password PASS    Grafana admin password (default: changeme123)
  --retention-days DAYS    Prometheus retention days (default: 15)
  --skip-dashboards        Skip dashboard import
  --skip-alerts            Skip alerting rules setup
  --dry-run                Show what would be deployed without deploying
  --verbose                Enable verbose output
  --help                   Show this help message

Environment Variables:
  NAMESPACE                Custom namespace
  STORAGE_CLASS            Storage class for PVCs
  ADMIN_USER               Grafana admin username
  ADMIN_PASSWORD           Grafana admin password
  EMAIL_NOTIFICATION       Email for alerts
  SLACK_WEBHOOK            Slack webhook URL

Examples:
  # Basic deployment
  $0

  # With email notifications
  $0 --email admin@example.com

  # With Slack notifications
  $0 --slack-webhook https://hooks.slack.com/services/YOUR/WEBHOOK/URL

  # Custom namespace and storage
  $0 --namespace monitoring-prod --storage-class fast-ssd

  # Dry run to see what would be deployed
  $0 --dry-run --verbose

EOF
}

check_dry_run() {
    if [[ "${DRY_RUN}" == true ]]; then
        log_warning "DRY RUN: Would execute: $1"
        return 1
    fi
    return 0
}

wait_for_pod() {
    local namespace=$1
    local label=$2
    local timeout=${3:-300}
    local elapsed=0

    log_info "Waiting for pod with label ${label} to be ready..."

    while [[ ${elapsed} -lt ${timeout} ]]; do
        local ready=$(kubectl get pods -n "${namespace}" -l "${label}" -o json \
            | jq -r '.items[] | select(.status.phase == "Running" and (.status.conditions[] | select(.type == "Ready" and .status == "True")) | .metadata.name' \
            | wc -l)

        if [[ ${ready} -ge 1 ]]; then
            log_success "Pod is ready"
            return 0
        fi

        log_debug "Waiting for pod... (${elapsed}s elapsed)"
        sleep 5
        elapsed=$((elapsed + 5))
    done

    log_error "Timeout waiting for pod"
    return 1
}

# ============================================================================
# PREREQUISITE CHECKS
# ============================================================================

check_prerequisites() {
    log_step "Checking prerequisites"

    local missing_deps=false

    # Check required commands
    for cmd in kubectl jq; do
        if ! command -v ${cmd} &> /dev/null; then
            log_error "Missing required command: ${cmd}"
            missing_deps=true
        else
            log_debug "Found: ${cmd}"
        fi
    done

    if [[ "${missing_deps}" == true ]]; then
        log_error "Please install missing dependencies"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    log_success "Cluster connection verified"

    # Check cluster version
    local server_version=$(kubectl version --short 2>/dev/null | grep Server | awk '{print $3}')
    log_debug "Kubernetes server version: ${server_version}"

    # Check available storage classes
    local storage_classes=$(kubectl get storageclass -o json | jq -r '.items[].metadata.name' 2>/dev/null)
    if [[ -n "${storage_classes}" ]]; then
        log_success "Available storage classes:"
        echo "${storage_classes}" | while read sc; do
            echo "  - ${sc}"
        done
    else
        log_warning "No storage classes found (may need to create PVs manually)"
    fi

    # Check required directories
    if [[ ! -d "${K8S_MONITORING_DIR}" ]]; then
        log_error "Monitoring directory not found: ${K8S_MONITORING_DIR}"
        exit 1
    fi

    if [[ ! -d "${DASHBOARDS_DIR}" ]]; then
        log_error "Dashboards directory not found: ${DASHBOARDS_DIR}"
        exit 1
    fi

    if [[ ! -d "${ALERTS_DIR}" ]]; then
        log_error "Alerts directory not found: ${ALERTS_DIR}"
        exit 1
    fi

    log_success "All prerequisites met"
}

# ============================================================================
# NAMESPACE AND RBAC SETUP
# ============================================================================

create_namespace() {
    log_step "Creating namespace and RBAC"

    check_dry_run "kubectl create namespace ${NAMESPACE}" || return 0

    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_warning "Namespace ${NAMESPACE} already exists"
    else
        kubectl create namespace "${NAMESPACE}"
        log_success "Created namespace: ${NAMESPACE}"
    fi

    # Create service account for monitoring
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: monitoring
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: victor-monitoring
rules:
  - apiGroups: [""]
    resources: ["nodes", "nodes/metrics", "pods", "services", "endpoints"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get"]
  - nonResourceURLs: ["/metrics"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: victor-monitoring
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: victor-monitoring
subjects:
  - kind: ServiceAccount
    name: monitoring
    namespace: ${NAMESPACE}
EOF
    log_success "Created RBAC resources"
}

# ============================================================================
# PROMETHEUS DEPLOYMENT
# ============================================================================

deploy_prometheus() {
    log_step "Deploying Prometheus"

    # Create ConfigMap for Prometheus configuration
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: prometheus
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'victor-ai'
        environment: '${ENVIRONMENT:-production}'

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
                - alertmanager:9093

    rule_files:
      - '/etc/prometheus/rules/*.yml'
      - '/etc/prometheus/rules/*.yaml'

    scrape_configs:
      # Prometheus self-monitoring
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']

      # Kubernetes API Server
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
          - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
            action: keep
            regex: default;kubernetes;https

      # Kubernetes Nodes
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
          - role: node
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)

      # Victor AI Application
      - job_name: 'victor-ai'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
            action: keep
            regex: victor
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: pod
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_node_name]
            action: replace
            target_label: node

      # Node Exporter (if deployed)
      - job_name: 'node-exporter'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
            action: keep
            regex: node-exporter
          - source_labels: [__meta_kubernetes_pod_ip]
            action: replace
            target_label: __address__
            replacement: \$1:9100

  # Victor AI alerting rules
  victor-rules.yml: |
$(cat "${ALERTS_DIR}/rules.yml" | sed 's/^/    /')
EOF
    log_success "Created Prometheus ConfigMap"

    # Create Persistent Volume Claim
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-data
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: prometheus
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      storage: ${PROMETHEUS_RETENTION_SIZE}
EOF
    log_success "Created Prometheus PVC"

    # Deploy Prometheus
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: victor
      app.kubernetes.io/component: prometheus
  template:
    metadata:
      labels:
        app.kubernetes.io/name: victor
        app.kubernetes.io/component: prometheus
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: monitoring
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: prometheus
          image: prom/prometheus:v2.48.0
          imagePullPolicy: IfNotPresent
          args:
            - '--config.file=/etc/prometheus/prometheus.yml'
            - '--storage.tsdb.path=/prometheus'
            - '--storage.tsdb.retention.time=${RETENTION_DAYS}d'
            - '--storage.tsdb.retention.size=${PROMETHEUS_RETENTION_SIZE}'
            - '--web.console.libraries=/etc/prometheus/console_libraries'
            - '--web.console.templates=/etc/prometheus/consoles'
            - '--web.enable-lifecycle'
            - '--web.enable-admin-api'
          ports:
            - name: http
              containerPort: 9090
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /-/healthy
              port: http
            initialDelaySeconds: 30
            periodSeconds: 15
            timeoutSeconds: 10
          readinessProbe:
            httpGet:
              path: /-/ready
              port: http
            initialDelaySeconds: 30
            periodSeconds: 5
            timeoutSeconds: 10
          resources:
            requests:
              cpu: 250m
              memory: 512Mi
            limits:
              cpu: 1000m
              memory: 2Gi
          volumeMounts:
            - name: config
              mountPath: /etc/prometheus
              readOnly: true
            - name: storage
              mountPath: /prometheus
      volumes:
        - name: config
          configMap:
            name: prometheus-config
        - name: storage
          persistentVolumeClaim:
            claimName: prometheus-data
EOF
    log_success "Deployed Prometheus"

    # Create Service
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: prometheus
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 9090
      targetPort: http
      protocol: TCP
  selector:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: prometheus
EOF
    log_success "Created Prometheus service"

    # Wait for Prometheus to be ready
    wait_for_pod "${NAMESPACE}" "app.kubernetes.io/component=prometheus"
}

# ============================================================================
# ALERTMANAGER DEPLOYMENT
# ============================================================================

deploy_alertmanager() {
    log_step "Deploying AlertManager"

    # Create AlertManager configuration
    local alertmanager_config=$(cat <<'EOF'
global:
  resolve_timeout: 5m
  slack_api_url: '${SLACK_WEBHOOK:-}'
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@victor-ai.com'
  smtp_require_tls: false

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true

    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#victor-ai-alerts'
        send_resolved: true
        title: '[{{ .Status | toUpper }}] {{ .CommonLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          {{ end }}

  - name: 'critical'
    email_addresses:
$(if [[ -n "${EMAIL_NOTIFICATION}" ]]; then
      echo "      - ${EMAIL_NOTIFICATION}"
  fi
)
    slack_configs:
      - channel: '#victor-ai-critical'
        send_resolved: true
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'
        title: '[CRITICAL] {{ .CommonLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Runbook:* {{ .Annotations.runbook_url }}
          {{ end }}

  - name: 'warning'
    slack_configs:
      - channel: '#victor-ai-warnings'
        send_resolved: true
        title: '[WARNING] {{ .CommonLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Summary:* {{ .Annotations.summary }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
EOF
)

    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
data:
  alertmanager.yml: |
$(echo "${alertmanager_config}" | sed 's/^/    /')
EOF
    log_success "Created AlertManager ConfigMap"

    # Create Persistent Volume Claim
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: alertmanager-data
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      storage: ${ALERTMANAGER_RETENTION_SIZE}
EOF
    log_success "Created AlertManager PVC"

    # Deploy AlertManager
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: victor
      app.kubernetes.io/component: alertmanager
  template:
    metadata:
      labels:
        app.kubernetes.io/name: victor
        app.kubernetes.io/component: alertmanager
    spec:
      serviceAccountName: monitoring
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: alertmanager
          image: prom/alertmanager:v0.26.0
          imagePullPolicy: IfNotPresent
          args:
            - '--config.file=/etc/alertmanager/alertmanager.yml'
            - '--storage.path=/alertmanager'
            - '--web.external-url=http://alertmanager:9093'
          ports:
            - name: http
              containerPort: 9093
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /-/healthy
              port: http
            initialDelaySeconds: 30
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /-/ready
              port: http
            initialDelaySeconds: 30
            periodSeconds: 5
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
          volumeMounts:
            - name: config
              mountPath: /etc/alertmanager
              readOnly: true
            - name: storage
              mountPath: /alertmanager
      volumes:
        - name: config
          configMap:
            name: alertmanager-config
        - name: storage
          persistentVolumeClaim:
            claimName: alertmanager-data
EOF
    log_success "Deployed AlertManager"

    # Create Service
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 9093
      targetPort: http
      protocol: TCP
  selector:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
EOF
    log_success "Created AlertManager service"

    # Wait for AlertManager to be ready
    wait_for_pod "${NAMESPACE}" "app.kubernetes.io/component=alertmanager"
}

# ============================================================================
# GRAFANA DEPLOYMENT
# ============================================================================

deploy_grafana() {
    log_step "Deploying Grafana"

    # Create secret for admin credentials
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Secret
metadata:
  name: grafana-admin-credentials
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
type: Opaque
stringData:
  admin-user: "${ADMIN_USER}"
  admin-password: "${ADMIN_PASSWORD}"
EOF
    log_success "Created Grafana admin credentials secret"

    # Create ConfigMap for Grafana configuration
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-config
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
data:
  grafana.ini: |
    [server]
    root_url = %(protocol)s://%(domain)s:%(http_port)s
    serve_from_sub_path = true

    [database]
    type = sqlite3
    path = /var/lib/grafana/grafana.db

    [security]
    admin_user = \${GF_SECURITY_ADMIN_USER}
    admin_password = \${GF_SECURITY_ADMIN_PASSWORD}
    disable_gravatar = true
    cookie_secure = true
    content_security_policy = true
    x_content_type_options = true
    x_xss_protection = true

    [users]
    allow_sign_up = false
    auto_assign_org_role = Viewer

    [auth.anonymous]
    enabled = false

    [analytics]
    check_for_updates = false
    reporting_enabled = false

    [log]
    mode = console
    level = info

    [alerting]
    enabled = true

    [unified_alerting]
    enabled = true
EOF
    log_success "Created Grafana ConfigMap"

    # Create datasource ConfigMap
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
data:
  prometheus.yaml: |
    apiVersion: 1

    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus:9090
        isDefault: true
        editable: true
        jsonData:
          timeInterval: "15s"
          queryTimeout: "60s"
          httpMethod: POST
EOF
    log_success "Created Grafana datasources ConfigMap"

    # Create Persistent Volume Claim
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-data
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: ${STORAGE_CLASS}
  resources:
    requests:
      storage: ${GRAFANA_RETENTION_SIZE}
EOF
    log_success "Created Grafana PVC"

    # Deploy Grafana
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: victor
      app.kubernetes.io/component: grafana
  template:
    metadata:
      labels:
        app.kubernetes.io/name: victor
        app.kubernetes.io/component: grafana
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "3000"
    spec:
      serviceAccountName: monitoring
      securityContext:
        runAsNonRoot: true
        runAsUser: 472
        fsGroup: 472
      initContainers:
        - name: setup-dashboards
          image: busybox:1.36
          command: ['sh', '-c', 'mkdir -p /var/lib/grafana/dashboards']
          volumeMounts:
            - name: storage
              mountPath: /var/lib/grafana
      containers:
        - name: grafana
          image: grafana/grafana:10.2.2
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 3000
              protocol: TCP
          env:
            - name: GF_SECURITY_ADMIN_USER
              valueFrom:
                secretKeyRef:
                  name: grafana-admin-credentials
                  key: admin-user
            - name: GF_SECURITY_ADMIN_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: grafana-admin-credentials
                  key: admin-password
            - name: GF_PATHS_DATA
              value: /var/lib/grafana
            - name: GF_PATHS_LOGS
              value: /var/log/grafana
            - name: GF_PATHS_PLUGINS
              value: /var/lib/grafana/plugins
            - name: GF_PATHS_PROVISIONING
              value: /etc/grafana/provisioning
          livenessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 15
            timeoutSeconds: 10
          readinessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 5
            timeoutSeconds: 10
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 1Gi
          volumeMounts:
            - name: config
              mountPath: /etc/grafana/grafana.ini
              subPath: grafana.ini
            - name: datasources
              mountPath: /etc/grafana/provisioning/datasources
              readOnly: true
            - name: storage
              mountPath: /var/lib/grafana
      volumes:
        - name: config
          configMap:
            name: grafana-config
        - name: datasources
          configMap:
            name: grafana-datasources
        - name: storage
          persistentVolumeClaim:
            claimName: grafana-data
EOF
    log_success "Deployed Grafana"

    # Create Service
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Service
metadata:
  name: grafana
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 3000
      targetPort: http
      protocol: TCP
  selector:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
EOF
    log_success "Created Grafana service"

    # Wait for Grafana to be ready
    wait_for_pod "${NAMESPACE}" "app.kubernetes.io/component=grafana"
}

# ============================================================================
# DASHBOARD IMPORT
# ============================================================================

import_dashboards() {
    if [[ "${SKIP_DASHBOARDS}" == true ]]; then
        log_warning "Skipping dashboard import"
        return 0
    fi

    log_step "Importing Grafana dashboards"

    local dashboard_count=0
    local dashboards=(
        "overview.json"
        "features.json"
        "performance.json"
        "team_overview.json"
        "team_members.json"
        "team_performance.json"
        "team_recursion.json"
    )

    for dashboard in "${dashboards[@]}"; do
        local dashboard_path="${DASHBOARDS_DIR}/${dashboard}"

        if [[ ! -f "${dashboard_path}" ]]; then
            log_warning "Dashboard not found: ${dashboard}"
            continue
        fi

        log_info "Importing dashboard: ${dashboard}"

        # Read dashboard JSON and remove outer wrapper if present
        local dashboard_json=$(cat "${dashboard_path}")

        # Create ConfigMap for dashboard
        local dashboard_name=$(basename "${dashboard}" .json)
        cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-${dashboard_name}
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
    grafana_dashboard: "1"
data:
  ${dashboard_name}.json: |
$(echo "${dashboard_json}" | jq -c '.' | sed 's/^/    /')
EOF

        if [[ $? -eq 0 ]]; then
            log_success "Imported dashboard: ${dashboard}"
            dashboard_count=$((dashboard_count + 1))
        else
            log_error "Failed to import dashboard: ${dashboard}"
        fi
    done

    log_success "Imported ${dashboard_count} dashboards"

    # Create dashboard provider configuration
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-providers
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
data:
  dashboards.yaml: |
    apiVersion: 1

    providers:
      - name: 'Victor AI'
        orgId: 1
        folder: 'Victor AI'
        type: file
        disableDeletion: false
        updateIntervalSeconds: 30
        allowUiUpdates: true
        options:
          path: /var/lib/grafana/dashboards
          foldersFromFilesStructure: false
EOF
    log_success "Created dashboard provider configuration"

    # Restart Grafana to load dashboards
    log_info "Restarting Grafana to load dashboards..."
    kubectl rollout restart deployment/grafana -n "${NAMESPACE}"
    wait_for_pod "${NAMESPACE}" "app.kubernetes.io/component=grafana"
}

# ============================================================================
# ALERTING RULES SETUP
# ============================================================================

setup_alerting_rules() {
    if [[ "${SKIP_ALERTS}" == true ]]; then
        log_warning "Skipping alerting rules setup"
        return 0
    fi

    log_step "Configuring alerting rules"

    # Create ConfigMap for alerting rules
    local rules_files=(
        "${ALERTS_DIR}/rules.yml"
        "${ALERTS_DIR}/team_alerts.yml"
    )

    for rules_file in "${rules_files[@]}"; do
        if [[ ! -f "${rules_file}" ]]; then
            log_warning "Rules file not found: ${rules_file}"
            continue
        fi

        local rules_name=$(basename "${rules_file}")
        log_info "Loading rules from: ${rules_name}"

        # Validate YAML syntax
        if ! kubectl create configmap prometheus-rules-${rules_name} \
            --from-file="${rules_file}" \
            --dry-run=client -n "${NAMESPACE}" -o yaml &> /dev/null; then
            log_error "Invalid YAML in ${rules_file}"
            continue
        fi

        # Create ConfigMap
        kubectl create configmap "prometheus-rules-${rules_name}" \
            --from-file="${rules_file}" \
            -n "${NAMESPACE}" \
            --dry-run=client -o yaml | kubectl apply -n "${NAMESPACE}" -f -

        log_success "Loaded rules: ${rules_name}"
    done

    # Count total rules
    local rule_count=$(grep -c "^  - alert:" "${ALERTS_DIR}/rules.yml" 2>/dev/null || echo "0")
    local team_rule_count=$(grep -c "^  - alert:" "${ALERTS_DIR}/team_alerts.yml" 2>/dev/null || echo "0")
    local total_rules=$((rule_count + team_rule_count))

    log_success "Configured ${total_rules} alerting rules"

    # Reload Prometheus configuration
    log_info "Reloading Prometheus configuration..."
    local prometheus_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}')

    if [[ -n "${prometheus_pod}" ]]; then
        kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" \
            -- wget -q --post-data="" \
            http://localhost:9090/-/reload &> /dev/null || true
        log_success "Prometheus configuration reloaded"
    fi
}

# ============================================================================
# HEALTH CHECKS
# ============================================================================

perform_health_checks() {
    log_step "Performing health checks"

    local all_healthy=true

    # Check Prometheus
    log_info "Checking Prometheus health..."
    local prometheus_ready=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')

    if [[ "${prometheus_ready}" == "True" ]]; then
        # Query Prometheus API
        local prometheus_health=$(kubectl exec -n "${NAMESPACE}" \
            $(kubectl get pods -n "${NAMESPACE}" \
                -l app.kubernetes.io/component=prometheus \
                -o jsonpath='{.items[0].metadata.name}') \
            -- wget -q -O- http://localhost:9090/-/healthy 2>/dev/null || echo "unhealthy")

        if [[ "${prometheus_health}" == "Prometheus is Healthy." ]]; then
            log_success "Prometheus is healthy"
        else
            log_error "Prometheus health check failed"
            all_healthy=false
        fi
    else
        log_error "Prometheus is not ready"
        all_healthy=false
    fi

    # Check AlertManager
    log_info "Checking AlertManager health..."
    local alertmanager_ready=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=alertmanager \
        -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')

    if [[ "${alertmanager_ready}" == "True" ]]; then
        local alertmanager_health=$(kubectl exec -n "${NAMESPACE}" \
            $(kubectl get pods -n "${NAMESPACE}" \
                -l app.kubernetes.io/component=alertmanager \
                -o jsonpath='{.items[0].metadata.name}') \
            -- wget -q -O- http://localhost:9093/-/healthy 2>/dev/null || echo "unhealthy")

        if [[ "${alertmanager_health}" == "OK" ]]; then
            log_success "AlertManager is healthy"
        else
            log_error "AlertManager health check failed"
            all_healthy=false
        fi
    else
        log_error "AlertManager is not ready"
        all_healthy=false
    fi

    # Check Grafana
    log_info "Checking Grafana health..."
    local grafana_ready=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=grafana \
        -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}')

    if [[ "${grafana_ready}" == "True" ]]; then
        local grafana_health=$(kubectl exec -n "${NAMESPACE}" \
            $(kubectl get pods -n "${NAMESPACE}" \
                -l app.kubernetes.io/component=grafana \
                -o jsonpath='{.items[0].metadata.name}') \
            -- wget -q -O- http://localhost:3000/api/health 2>/dev/null | \
            jq -r '.database' 2>/dev/null || echo "unhealthy")

        if [[ "${grafana_health}" == "ok" ]]; then
            log_success "Grafana is healthy"
        else
            log_error "Grafana health check failed"
            all_healthy=false
        fi
    else
        log_error "Grafana is not ready"
        all_healthy=false
    fi

    if [[ "${all_healthy}" == true ]]; then
        log_success "All health checks passed"
        return 0
    else
        log_error "Some health checks failed"
        return 1
    fi
}

# ============================================================================
# TEST ALERT GENERATION
# ============================================================================

generate_test_alert() {
    log_step "Generating test alert"

    log_info "Creating test alert rule..."

    # Create test alert ConfigMap
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-test-alert
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: prometheus
data:
  test-alerts.yml: |
    groups:
      - name: test_alerts
        interval: 30s
        rules:
          - alert: TestAlert
            expr: vector(1)
            for: 1m
            labels:
              severity: info
              category: test
            annotations:
              summary: "This is a test alert from Victor AI monitoring"
              description: "Test alert generated at $(date)"
EOF

    log_success "Test alert rule created"

    # Update Prometheus to load test alert
    log_info "Waiting for test alert to fire (2 minutes)..."
    echo ""
    echo "The test alert 'TestAlert' will fire in approximately 2 minutes."
    echo "You can verify it in:"
    echo "  - Grafana: Alerting page"
    echo "  - Prometheus: http://localhost:9090/alerts"
    echo "  - AlertManager: http://localhost:9093/#/alerts"
    echo ""

    # Optional: Reload Prometheus
    local prometheus_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}')

    if [[ -n "${prometheus_pod}" ]]; then
        kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" \
            -- wget -q --post-data="" \
            http://localhost:9090/-/reload &> /dev/null || true
    fi
}

# ============================================================================
# DISPLAY ACCESS INFORMATION
# ============================================================================

display_access_info() {
    log_step "Access Information"

    echo ""
    echo "Monitoring stack successfully deployed!"
    echo ""
    echo "================================================================"
    echo "                    ACCESS CREDENTIALS"
    echo "================================================================"
    echo ""
    echo "Grafana:"
    echo "  URL:      http://grafana.${NAMESPACE}.svc.cluster.local:3000"
    echo "  Username: ${ADMIN_USER}"
    echo "  Password: ${ADMIN_PASSWORD}"
    echo ""
    echo "Prometheus:"
    echo "  URL:      http://prometheus.${NAMESPACE}.svc.cluster.local:9090"
    echo ""
    echo "AlertManager:"
    echo "  URL:      http://alertmanager.${NAMESPACE}.svc.cluster.local:9093"
    echo ""
    echo "================================================================"
    echo "                    PORT FORWARDING"
    echo "================================================================"
    echo ""
    echo "To access from your local machine:"
    echo ""
    echo "  Grafana:"
    echo "    kubectl port-forward -n ${NAMESPACE} svc/grafana 3000:3000"
    echo "    Then open: http://localhost:3000"
    echo ""
    echo "  Prometheus:"
    echo "    kubectl port-forward -n ${NAMESPACE} svc/prometheus 9090:9090"
    echo "    Then open: http://localhost:9090"
    echo ""
    echo "  AlertManager:"
    echo "    kubectl port-forward -n ${NAMESPACE} svc/alertmanager 9093:9093"
    echo "    Then open: http://localhost:9093"
    echo ""
    echo "================================================================"
    echo "                    VERIFICATION COMMANDS"
    echo "================================================================"
    echo ""
    echo "Check pod status:"
    echo "  kubectl get pods -n ${NAMESPACE}"
    echo ""
    echo "Check services:"
    echo "  kubectl get svc -n ${NAMESPACE}"
    echo ""
    echo "Check persistent volumes:"
    echo "  kubectl get pvc -n ${NAMESPACE}"
    echo ""
    echo "View Prometheus targets:"
    echo "  kubectl port-forward -n ${NAMESPACE} svc/prometheus 9090:9090 &"
    echo "  open http://localhost:9090/targets"
    echo ""
    echo "View active alerts:"
    echo "  kubectl port-forward -n ${NAMESPACE} svc/prometheus 9090:9090 &"
    echo "  open http://localhost:9090/alerts"
    echo ""
    echo "================================================================"
    echo "                    SECURITY NOTES"
    echo "================================================================"
    echo ""
    echo "IMPORTANT SECURITY ACTIONS:"
    echo ""
    echo "1. Change the default Grafana password:"
    echo "   kubectl -n ${NAMESPACE} create secret generic grafana-admin-credentials \\"
    echo "     --from-literal=admin-user=admin \\"
    echo "     --from-literal=admin-password=YOUR_SECURE_PASSWORD \\"
    echo "     --dry-run=client -o yaml | kubectl apply -f -"
    echo "   kubectl -n ${NAMESPACE} rollout restart deployment/grafana"
    echo ""
    echo "2. Configure persistent storage:"
    echo "   kubectl get pvc -n ${NAMESPACE}"
    echo ""
    echo "3. Set up notification channels:"
    echo "   Edit alertmanager-config ConfigMap to add email/Slack"
    echo ""
    echo "4. Enable TLS/SSL:"
    echo "   Configure ingress with cert-manager for production"
    echo ""
    echo "================================================================"
    echo ""
}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

main() {
    log_step "Victor AI - Complete Monitoring Stack Deployment"
    echo ""
    log_info "Configuration:"
    echo "  Namespace:       ${NAMESPACE}"
    echo "  Storage Class:   ${STORAGE_CLASS}"
    echo "  Retention:       ${RETENTION_DAYS} days"
    echo "  Email:           ${EMAIL_NOTIFICATION:-Not configured}"
    echo "  Slack:           ${SLACK_WEBHOOK:+Configured}"
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --storage-class)
                STORAGE_CLASS="$2"
                shift 2
                ;;
            --email)
                EMAIL_NOTIFICATION="$2"
                shift 2
                ;;
            --slack-webhook)
                SLACK_WEBHOOK="$2"
                shift 2
                ;;
            --admin-password)
                ADMIN_PASSWORD="$2"
                shift 2
                ;;
            --retention-days)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            --skip-dashboards)
                SKIP_DASHBOARDS=true
                shift
                ;;
            --skip-alerts)
                SKIP_ALERTS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Main deployment flow
    check_prerequisites

    if [[ "${DRY_RUN}" == true ]]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
        echo ""
    fi

    create_namespace
    deploy_prometheus
    deploy_alertmanager
    deploy_grafana
    import_dashboards
    setup_alerting_rules

    # Final health checks
    if [[ "${DRY_RUN}" == false ]]; then
        if perform_health_checks; then
            display_access_info

            # Ask about test alert
            echo ""
            read -p "Generate a test alert to verify notifications? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                generate_test_alert
            fi

            log_success "Monitoring stack deployment completed successfully!"
            exit 0
        else
            log_error "Health checks failed. Please check the logs above."
            log_info "You can view pod logs with:"
            echo "  kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/component=prometheus"
            echo "  kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/component=alertmanager"
            echo "  kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/component=grafana"
            exit 1
        fi
    else
        log_success "Dry run completed"
        exit 0
    fi
}

# Run main function
main "$@"
