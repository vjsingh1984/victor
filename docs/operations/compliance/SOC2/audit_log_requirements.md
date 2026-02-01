# SOC2 Audit Log Requirements - Victor AI

> **Template**: This document describes intended controls. It does not assert current certification or compliance. Update with actual audit evidence and operational details.


**Version:** 1.0
**Last Updated:** 2026-01-20
**Next Review:** 2026-04-20
**Owner:** DevOps Lead

---

## Table of Contents

1. [Overview](#1-overview)
2. [Logging Requirements](#2-logging-requirements)
3. [Event Types to Log](#3-event-types-to-log)
4. [Log Format and Structure](#4-log-format-and-structure)
5. [Log Retention](#5-log-retention)
6. [Log Protection](#6-log-protection)
7. [Log Monitoring and Review](#7-log-monitoring-and-review)
8. [Audit Evidence Collection](#8-audit-evidence-collection)

---

## 1. Overview

### 1.1 Purpose

This document defines the audit logging requirements for Victor AI in accordance with SOC2 Trust Services Criteria (CC3.3, CC6.7, CC7.2).

### 1.2 Scope

This requirement applies to:
- All systems (applications, databases, infrastructure)
- All user activities (administrative, operational)
- All security events
- All system changes

### 1.3 Objectives

- Maintain comprehensive audit trail
- Support forensic investigations
- Enable compliance verification
- Detect and investigate security incidents

---

## 2. Logging Requirements

### 2.1 Must-Log Events (SOC2 Required)

**Authentication and Access:**
- All login attempts (success and failure)
- Logout events
- Privileged access use
- MFA challenges
- Account lockouts
- Password changes/resets
- Access authorization failures

**System Changes:**
- Configuration changes
- Software installations/updates
- User account changes (create, modify, delete)
- Permission changes
- Firewall rule changes
- Network configuration changes

**Data Access:**
- Access to customer data
- Access to confidential data
- Database queries (for sensitive data)
- File access (for sensitive files)
- Data export/transfer

**Administrative Actions:**
- System administration activities
- Privileged command execution
- Security policy modifications
- Audit log access/changes

**Security Events:**
- Malware detection
- Intrusion detection alerts
- Anomalous activity detection
- DLP alerts (if implemented)
- Firewall block events

**Application Events:**
- Application errors
- API access
- Feature usage
- Business transactions

### 2.2 Logging Infrastructure

**Log Sources:**

| Source | Events Logged | Format |
|--------|---------------|--------|
| **Application** | User actions, errors, API calls | JSON |
| **Authentication** | Logins, MFA, authorization | JSON |
| **Database** | Queries, schema changes, access | Query log, binlog |
| **Web Server** | HTTP requests, errors | Combined log format |
| **Load Balancer** | Requests, TLS termination | Custom format |
| **Operating System** | System events, auth, sudo | Syslog |
| **Cloud Infrastructure (AWS)** | API calls, console login | CloudTrail |
| **Container Runtime (Kubernetes)** | Pod events, API calls | Audit log |
| **Network** | Firewall rules, flows | NetFlow, Syslog |

**Log Aggregation:**
- Centralized logging (ELK Stack, Splunk, or CloudWatch)
- Real-time log streaming from all sources
- Log buffering for network interruptions
- Time synchronization (NTP)

---

## 3. Event Types to Log

### 3.1 Detailed Event Catalog

**Authentication Events:**

| Event | Fields Required | Example |
|-------|-----------------|---------|
| Login attempt | timestamp, user_id, source_ip, result, method | User jane.doe logged in from 192.168.1.1 (success) |
| Login failure | timestamp, user_id, source_ip, failure_reason | User john.doe failed login (invalid password) |
| MFA challenge | timestamp, user_id, mfa_method, result | User jane.doe MFA challenge (TOTP) success |
| Password change | timestamp, user_id, initiator, source_ip | User jane.doe changed password |
| Account lockout | timestamp, user_id, reason, locked_until | User john.doe locked (too many failed attempts) |

**Authorization Events:**

| Event | Fields Required | Example |
|-------|-----------------|---------|
| Permission granted | timestamp, user_id, permission, granter | User jane.doe granted admin access to database |
| Permission revoked | timestamp, user_id, permission, revoker | User john.doe revoked from admin role |
| Access denied | timestamp, user_id, resource, reason | User guest denied access to /admin (insufficient permissions) |

**Data Access Events:**

| Event | Fields Required | Example |
|-------|-----------------|---------|
| Customer data access | timestamp, user_id, customer_id, data_type, action | User support.viewed customer data for customer_123 |
| Data export | timestamp, user_id, record_count, format, destination | User analyst exported 1000 records to CSV |
| Database query | timestamp, user_id, query_hash, tables_affected, rows | User app ran SELECT on users table (50 rows) |

**System Change Events:**

| Event | Fields Required | Example |
|-------|-----------------|---------|
| Configuration change | timestamp, user_id, config_file, change_type, old_value, new_value | User admin changed log_level from INFO to DEBUG |
| Software deployment | timestamp, deployer, version, environment, result | User devops deployed v1.2.3 to production (success) |
| Firewall rule change | timestamp, admin, rule_id, action, change | User admin added firewall rule allowing port 443 |

**Security Events:**

| Event | Fields Required | Example |
|-------|-----------------|---------|
| Malware detected | timestamp, host, malware_type, file, action | Malware Trojan.Generic detected on server-1 (quarantined) |
| Intrusion detected | timestamp, source_ip, target, attack_type, action | SQL injection attempt blocked from 192.168.1.100 |
| Anomaly detected | timestamp, anomaly_type, severity, description | Unusual data transfer detected (10GB upload) |

### 3.2 Event Priority

| Priority | Event Types | Retention |
|----------|-------------|-----------|
| **Critical** | Security incidents, data breaches, system compromises | 7 years |
| **High** | Authentication failures, authorization failures, data access | 1 year |
| **Medium** | Configuration changes, deployments, errors | 90 days |
| **Low** | Normal operations, performance metrics | 30 days |

---

## 4. Log Format and Structure

### 4.1 Standard Log Format

**JSON Structure (Recommended):**

```json
{
  "timestamp": "2026-01-20T14:30:00Z",
  "event_id": "evt_abc123",
  "event_type": "user_login",
  "event_source": "authentication_service",
  "event_category": "authentication",
  "severity": "info",
  "actor": {
    "user_id": "user_123",
    "username": "jane.doe@victor.ai",
    "role": "engineer",
    "session_id": "sess_xyz789"
  },
  "action": {
    "type": "login",
    "method": "password",
    "mfa_used": true,
    "mfa_method": "totp",
    "result": "success"
  },
  "resource": {
    "type": "application",
    "id": "victor_ai",
    "name": "Victor AI Application"
  },
  "network": {
    "source_ip": "192.168.1.100",
    "source_port": 54321,
    "destination_ip": "10.0.0.1",
    "destination_port": 443,
    "user_agent": "Mozilla/5.0..."
  },
  "location": {
    "country": "US",
    "region": "California",
    "city": "San Francisco"
  },
  "metadata": {
    "correlation_id": "corr_123",
    "request_id": "req_456",
    "environment": "production"
  }
}
```

### 4.2 Required Fields

**All logs MUST include:**
- **timestamp:** ISO 8601 format (UTC)
- **event_type:** Category of event
- **event_source:** System or service generating event
- **severity:** critical, error, warning, info, debug
- **actor:** User or system performing action (if applicable)
- **action:** What occurred
- **result:** success, failure, partial (if applicable)

**Additional fields as appropriate:**
- **network:** Source/destination IP and port
- **resource:** Target of action
- **changes:** Old and new values for changes
- **reason:** Justification for actions
- **correlation_id:** Link related events

### 4.3 Time Synchronization

**Requirements:**
- All systems synchronized to UTC
- NTP configuration on all servers
- Time drift tolerance: ±1 second
- Monitoring for time sync issues

**Implementation:**
- Kubernetes nodes: Chrony or systemd-timesyncd
- Cloud resources: Use cloud provider time service
- Monitoring: Alert if time drift > 1 second

---

## 5. Log Retention

### 5.1 Retention Periods

| Log Type | Hot Storage | Cold Storage | Archive | Total Retention |
|----------|-------------|--------------|---------|-----------------|
| **Security Events** | 90 days | - | 7 years | 7 years |
| **Audit Logs (Access)** | 90 days | 1 year | - | 1 year |
| **Audit Logs (Changes)** | 90 days | 1 year | - | 1 year |
| **Transaction Logs** | 90 days | - | - | 90 days |
| **Application Logs** | 30 days | - | - | 30 days |
| **Performance Logs** | 30 days | - | - | 30 days |

**Rationale:**
- **7 years:** SOC2 requirement for security-relevant logs
- **1 year:** Industry standard for audit logs
- **90 days:** SOC2 monitoring requirement
- **30 days:** Operational need

### 5.2 Storage Tiers

**Hot Storage (90 days):**
- Fast access for investigations
- Real-time monitoring and alerting
- SIEM indexing
- Cost: Higher (Elasticsearch, CloudWatch Logs)

**Cold Storage (1 year):**
- Slower access for compliance
- Compressed storage
- Cost: Lower (S3 Glacier, S3 Standard-IA)

**Archive (7 years):**
- Long-term compliance retention
- Very slow access
- Immutable storage (WORM)
- Cost: Lowest (S3 Glacier Deep Archive, tape)

### 5.3 Log Archival

**Process:**
1. **Retention Policy Check:** Daily check for logs exceeding hot storage period
2. **Move to Cold Storage:** Compress and move to cold storage
3. **Archive:** After 1 year, move to archive storage
4. **Verification:** Verify archival integrity
5. **Deletion:** After retention period, securely delete

**Archive Requirements:**
- Immutable (append-only, no modifications)
- Encrypted at rest
- Tamper-evident (hash verification)
- Access logged

---

## 6. Log Protection

### 6.1 Access Control

**Access to Logs:**
- **Security Team:** Full access for investigations
- **System Administrators:** Limited access (for troubleshooting)
- **Auditors:** Read-only access during audit
- **All Others:** No access (unless required for job function)

**Authorization:**
- Role-based access control
- Multi-factor authentication required
- Justification required for access
- All access logged

### 6.2 Integrity Protection

**Requirements:**
- Logs cannot be modified after creation
- Hash chains or digital signatures for integrity verification
- Tamper-evident storage
- Regular integrity verification

**Implementation:**
- Write-once-read-many (WORM) storage for archived logs
- Hash chaining (each log entry includes hash of previous)
- Regular integrity scans (daily)
- Alert on integrity violations

### 6.3 Encryption

**At Rest:**
- Hot storage: Encrypted (AES-256)
- Cold storage: Encrypted (AES-256)
- Archive: Encrypted (AES-256)
- Key rotation: Annual

**In Transit:**
- TLS 1.3 for log shipping
- Certificate-based authentication
- Certificate rotation: 90 days

**Key Management:**
- Keys stored in secure key management service
- Separate keys for each storage tier
- Key access logged and restricted

---

## 7. Log Monitoring and Review

### 7.1 Real-Time Monitoring

**Automated Alerts:**

| Event | Alert Threshold | Escalation |
|-------|----------------|------------|
| Multiple failed logins | 5 failures from same user in 5 minutes | Security team |
| Privileged access use | Any use outside business hours | Security team + manager |
| Data export | Any export of >1000 records | Security team |
| Security event | Any critical or high severity | Immediate notification |
| System change | Any change to production | Change Manager |
| Log integrity violation | Any tampering detected | CISO + CTO |

**Monitoring Tools:**
- SIEM (Elastic Security, Splunk, or Cloud-native)
- Real-time alerting (PagerDuty, Slack, email)
- Dashboard for security operations

### 7.2 Regular Review

**Daily Review (Security Team):**
- Critical and high alerts
- Overnight security events
- Anomalous activity

**Weekly Review (Security Team):**
- Trends and patterns
- Medium and low alerts
- False positive tuning

**Monthly Review (Security + Management):**
- Compliance status
- Risk assessment
- Process improvement

**Quarterly Review (CISO + Executive Team):**
- Log retention compliance
- Security posture
- Incident trends
- Recommendations

### 7.3 Log Analysis

**Trend Analysis:**
- Failed login trends
- Data access patterns
- Privileged access trends
- Security event trends

**Anomaly Detection:**
- Unusual time of access
- Unusual location of access
- Unusual data access volume
- Unusual system changes

**Reporting:**
- Monthly security report
- Quarterly compliance report
- Annual summary for audit

---

## 8. Audit Evidence Collection

### 8.1 Evidence for SOC2 Audit

**Required Evidence:**

| Evidence Category | Source | Retention |
|-------------------|--------|-----------|
| **Access logs** | Authentication system, application | 1 year |
| **Change logs** | Change management system, git | 1 year |
| **Security event logs** | SIEM, security tools | 7 years |
| **System logs** | Application, infrastructure | 90 days |
| **Network logs** | Firewall, load balancer | 90 days |
| **User activity logs** | Application | 1 year |

### 8.2 Evidence Collection Process

**For SOC2 Audit:**

1. **Pre-Audit Preparation (60 days before):**
   - Verify log retention requirements met
   - Test log retrieval process
   - Verify log integrity
   - Prepare evidence inventory

2. **Evidence Collection (During Audit):**
   - Export logs for audit period (typically 6-12 months)
   - Verify integrity (hash verification)
   - Provide to auditor via secure method
   - Document chain of custody

3. **Post-Audit:**
   - Archive audit evidence
   - Document auditor findings
   - Address gaps identified

### 8.3 Evidence Format

**Deliverable Formats:**
- Raw logs (JSON, CSV)
- Aggregated reports (PDF, Excel)
- Dashboards (read-only access)
- Custom exports (as requested by auditor)

**Secure Delivery:**
- Encrypted file transfer (SFTP, secure file sharing)
- Password protection + separate delivery of password
- Access expiration (30 days)
- Transfer logged

---

## 9. Compliance Verification

### 9.1 Logging Controls Testing

**Quarterly Tests:**
- Verify logging enabled on all systems
- Test log retrieval process
- Verify log integrity (hash verification)
- Test alerting mechanism
- Verify retention policy compliance

**Annual Tests:**
- Full audit trail simulation
- Evidence collection drill
- Retention period verification
- Disaster recovery test for logs

### 9.2 Compliance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Systems logging | 100% | Automated scan |
| Log completeness | > 99% | Time series analysis |
| Alert response time | < 15 minutes | Alert tracking |
| Log integrity violations | 0 | Integrity verification |
| Retention compliance | 100% | Automated checks |

---

## 10. Related Documents

- [SOC2 Security Policies](./policies.md)
- [SOC2 Access Control Procedures](./access_control.md)
- [SOC2 Incident Response Procedures](./incident_response.md)
- [SOC2 Change Management Procedures](./change_management.md)
- [SOC2 Data Classification Schema](./data_classification.md)
- [SOC2 Checklist](./soc2_checklist.md)

---

**END OF DOCUMENT**

---

**Last Updated:** February 01, 2026
**Reading Time:** 11 minutes
