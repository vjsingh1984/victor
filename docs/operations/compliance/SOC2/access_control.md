# SOC2 Access Control Procedures - Victor AI

> **Template**: This document describes intended controls. It does not assert current certification or compliance. Update with actual audit evidence and operational details.


**Version:** 1.0
**Last Updated:** 2026-01-20
**Next Review:** 2026-04-20
**Owner:** DevOps Lead

---

## Table of Contents

1. [Overview](#1-overview)
2. [Access Control Principles](#2-access-control-principles)
3. [Access Provisioning](#3-access-provisioning)
4. [Access Management](#4-access-management)
5. [Access Reviews](#5-access-reviews)
6. [Privileged Access Management](#6-privileged-access-management)
7. [Authentication](#7-authentication)
8. [Authorization](#8-authorization)
9. [Access Revocation](#9-access-revocation)

---

## 1. Overview

### 1.1 Purpose

This document defines the procedures for managing access to Victor AI systems, data, and facilities in accordance with SOC2 Trust Services Criteria (CC6.1, CC6.7, CC6.8).

### 1.2 Scope

This procedure applies to:
- All personnel (employees, contractors, interns)
- All systems (applications, databases, infrastructure)
- All facilities (offices, data centers)
- All access types (physical, logical, administrative)

### 1.3 Objectives

- Ensure only authorized access to information assets
- Implement principle of least privilege
- Maintain audit trail of all access changes
- Conduct regular access reviews

---

## 2. Access Control Principles

### 2.1 Principle of Least Privilege

Users are granted the minimum level of access required to perform their job functions.

**Implementation:**
- Role-based access control (RBAC)
- Need-to-know basis for sensitive data
- Time-limited access where appropriate
- Just-in-time access for privileged operations

### 2.2 Need-to-Know

Access to sensitive information is granted only to individuals with a documented business need.

**Criteria:**
- Job function requires access
- Access documented in job description
- Approved by manager and data owner

### 2.3 Separation of Duties

Critical functions require multiple individuals to prevent fraud and error.

**Examples:**
- Code changes require approval from separate reviewer
- Production deployments require approval from separate engineer
- Financial transactions require dual authorization

---

## 3. Access Provisioning

### 3.1 New User Access Request

**Trigger:**
- New employee hire
- Contractor engagement
- Intern assignment

**Process:**

1. **Manager Submits Request:**
   - Complete access request form
   - Specify job role and required access
   - Justify access to sensitive systems
   - Include start date and duration (if contractor)

2. **HR Approval:**
   - Verify employment/contract
   - Confirm job classification
   - Approve request

3. **IT Security Review:**
   - Verify access level matches job role
   - Check for conflicts of interest
   - Approve or request modification

4. **Access Provisioning:**
   - Create user accounts in systems
   - Assign appropriate roles/groups
   - Send temporary credentials to personal email
   - Require password change on first login

5. **Documentation:**
   - Log in access management system
   - Send confirmation to manager
   - Schedule access review (90 days)

**Timeline:** Within 24 hours of manager request

### 3.2 Access Request Template

```yaml
access_request:
  request_id: "AR-2026-001"
  request_date: "2026-01-20"
  requester: "Jane Smith (Engineering Manager)"
  user_details:
    name: "John Doe"
    email: "john.doe@victor.ai"
    employee_id: "EMP-2026-001"
    employment_type: "full_time"  # full_time, contractor, intern
    start_date: "2026-02-01"

  requested_access:
    systems:
      - name: "GitHub"
        role: "developer"
        justification: "Code repository access"
      - name: "AWS Console"
        role: "readonly"
        justification: "View cloud infrastructure"
      - name: "Jira"
        role: "developer"
        justification: "Project tracking"

    data_access:
      - classification: "internal"
        systems: ["confluence", "jira"]
      - classification: "confidential"
        systems: []
        justification: "N/A"

    privileged_access:
      required: false
      justification: "N/A"

  manager_approval:
    name: "Jane Smith"
    date: "2026-01-20"
    signature: "jane.smith@victor.ai"

  hr_approval:
    name: "HR Representative"
    date: "2026-01-20"
    signature: "hr@victor.ai"

  security_approval:
    name: "Security Lead"
    date: "2026-01-20"
    signature: "security@victor.ai"
```

---

## 4. Access Management

### 4.1 Role-Based Access Control (RBAC)

**Defined Roles:**

| Role | Description | Access Level | Systems |
|------|-------------|--------------|---------|
| **Admin** | Full system administration | High | All systems |
| **Engineer** | Software development | Medium | GitHub, CI/CD, Staging |
| **DevOps** | Infrastructure management | High | AWS, Kubernetes, Monitoring |
| **Support** | Customer support | Low | Support systems, Documentation |
| **Sales** | Sales and marketing | Low | CRM, Marketing tools |
| **Executive** | Company leadership | Medium | Reporting, Financial systems |
| **Contractor** | Third-party contractors | Variable | As specified |

### 4.2 Access Levels

| Access Level | Description | Examples |
|--------------|-------------|----------|
| **No Access** | No system access | N/A |
| **Read-Only** | View only, no modifications | Reporting, Dashboards |
| **Standard** | Normal user access | Email, Collaboration tools |
| **Power** | Advanced features | Development tools, Source control |
| **Administrative** | Full control | System administration, Infrastructure |

### 4.3 System-Specific Access

**GitHub:**
- Roles: Admin, Maintain, Write, Read, None
- Protected branches require approval
- Production deployment requires additional authorization

**AWS:**
- IAM roles with least privilege
- MFA required for console access
- API keys rotated every 90 days
- No root account usage

**Kubernetes:**
- RBAC configured per namespace
- Service accounts for applications
- No access to production for developers
- Separate admin accounts for cluster operations

**Databases:**
- Separate credentials per application
- Read replicas for read-only access
- No direct database access for analysts
- Connection pooling and rate limiting

---

## 5. Access Reviews

### 5.1 Quarterly Access Review

**Purpose:** Verify all access is still required and appropriate

**Scope:** All user access across all systems

**Process:**

1. **Generate Access Report:**
   - List all users and their access
   - Include access date and justification
   - Flag users with no recent activity

2. **Manager Review:**
   - Send report to each manager
   - Request confirmation of each access right
   - Flag unnecessary access for removal

3. **Data Owner Review:**
   - Send report to data owners
   - Review access to sensitive data
   - Approve or request removal

4. **Execute Changes:**
   - Remove unnecessary access
   - Document changes
   - Notify affected users

**Timeline:** Q1 (Jan), Q2 (Apr), Q3 (Jul), Q4 (Oct)

### 5.2 Annual Access Certification

**Purpose:** Formal certification of access by management

**Process:**

1. **Generate Certification Package:**
   - Complete access inventory
   - Include user roles and justification
   - Highlight high-risk access

2. **Management Certification:**
   - Each manager certifies their team's access
   - Exceptions documented and approved
   - Non-certified access removed

3. **Executive Sign-Off:**
   - CTO certifies technical access
   - CEO certifies overall program

**Timeline:** January each year

### 5.3 Event-Driven Reviews

Access is reviewed immediately when:
- Role change
- Department transfer
- Project completion
- Security incident involving access
- Policy violation

---

## 6. Privileged Access Management

### 6.1 Privileged Roles

Privileged roles include:
- System administrators
- Database administrators
- Cloud infrastructure administrators
- Security administrators
- Network administrators
- DevOps engineers

### 6.2 Privileged Access Requirements

**Pre-Approval:**
- Documented business justification
- Manager approval
- Security team approval
- Background check (for employees)

**Access Controls:**
- MFA required (hardware token preferred)
- Separate privileged account
- Session recording for audit
- Time-limited access where possible
- Just-in-time access for sensitive operations

**Monitoring:**
- All privileged access logged
- Real-time alerting for anomalous behavior
- Quarterly review of privileged access
- Annual recertification

### 6.3 Privileged Session Management

**Requirements:**
- Session timeout: 15 minutes idle
- Maximum session duration: 4 hours
- Concurrent sessions: Maximum 1
- Session recording enabled
- Audit logging of all commands

**Tools:**
- Teleport for SSH access
- AWS Systems Manager Session Manager
- Kubernetes audit logging

### 6.4 Emergency Access

**Break Glass Procedure:**

Access to privileged accounts in emergency:

1. **Initiation:**
   - Document emergency situation
   - Approve by CISO or CTO
   - Log access request

2. **Access:**
   - Use emergency account
   - Enable additional privileges
   - Record all actions

3. **Post-Emergency:**
   - Revoke emergency access
   - Document actions taken
   - Review within 24 hours
   - Implement preventive measures

**Use Cases:**
- System outage
- Security incident response
- Critical production issue

---

## 7. Authentication

### 7.1 Authentication Factors

**Standard Authentication:**
- Username + password
- Password complexity: 12+ characters, mixed types

**Multi-Factor Authentication (MFA):**
- Required for:
  - All administrative access
  - Remote access (VPN)
  - Access to customer data
  - Access from untrusted networks

- Methods:
  - TOTP app (Google Authenticator, Authy)
  - Hardware token (YubiKey) - preferred for admins
  - SMS - least preferred, allowed for standard users

### 7.2 Password Standards

**Creation:**
- Minimum 12 characters
- Complexity: uppercase, lowercase, numbers, special characters
- No common passwords (check against breached password lists)
- No personal information
- Unique to Victor AI

**Rotation:**
- No forced expiration for standard users (NIST 800-63B)
- Rotation required for:
  - Service accounts: 90 days
  - Admin accounts: Upon suspicion of compromise
  - All users: After security incident

**Reset:**
- Self-service password reset
- Temporary passwords expire in 24 hours
- Must change on first login
- Email notification of password change

### 7.3 Account Lockout

**Failed Login Attempts:**
- Lock after 5 failed attempts
- Lock duration: 15 minutes
- Admin unlock or self-service unlock

**Exceptions:**
- Service accounts: No lockout (use rate limiting)
- MFA bypass: Not allowed

### 7.4 Session Management

**Timeouts:**
- Idle timeout: 15 minutes
- Maximum session: 8 hours
- Forced reauthentication: Daily for admins

**Concurrent Sessions:**
- Standard users: Maximum 3
- Administrators: Maximum 1

**Session Termination:**
- User logout
- Session timeout
- Admin termination
- Account revocation

---

## 8. Authorization

### 8.1 Authorization Model

**Role-Based Access Control (RBAC):**
- Users assigned to roles
- Roles assigned permissions
- Permissions grant access to resources

**Implementation:**
```python
# Example RBAC implementation
from enum import Enum
from typing import Set, List

class Role(Enum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    DEVOPS = "devops"
    SUPPORT = "support"
    SALES = "sales"
    CONTRACTOR = "contractor"

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    DEPLOY = "deploy"

ROLE_PERMISSIONS = {
    Role.ADMIN: {Permission.ADMIN, Permission.READ, Permission.WRITE, Permission.DELETE, Permission.DEPLOY},
    Role.ENGINEER: {Permission.READ, Permission.WRITE},
    Role.DEVOPS: {Permission.READ, Permission.WRITE, Permission.DEPLOY, Permission.ADMIN},
    Role.SUPPORT: {Permission.READ},
    Role.SALES: {Permission.READ},
    Role.CONTRACTOR: {Permission.READ},
}

def check_permission(user_role: Role, required_permission: Permission) -> bool:
    """Check if user role has required permission"""
    return required_permission in ROLE_PERMISSIONS.get(user_role, set())
```

### 8.2 Resource-Level Authorization

**GitHub:**
- Repository access based on team membership
- Protected branches require pull request review
- Production deployment requires additional approval

**AWS:**
- IAM policies scoped to specific resources
- S3 bucket policies restrict access
- EC2 instance roles limit permissions

**Kubernetes:**
- Namespace-based isolation
- Role-based access control per namespace
- Network policies restrict communication

**API:**
- OAuth2 / JWT tokens for authentication
- Scope-based authorization
- Rate limiting per user

### 8.3 Data Access Authorization

**Customer Data:**
- Access only to assigned customers
- Data anonymization for analytics
- Audit logging of all access

**Internal Data:**
- Role-based access to reports
- Manager access to team data
- Executive access to all data

**Source Code:**
- All engineers have access
- Contractors limited to specific repos
- No access to production secrets

---

## 9. Access Revocation

### 9.1 Voluntary Termination (Resignation)

**Timeline:**
- Notice received: Immediately disable shared accounts
- Last day: Revoke all access
- 5 days post-employment: Forward email terminated

**Process:**

1. **Notification to IT:**
   - HR notifies IT of resignation
   - Provide last day of work
   - Specify if email forwarding needed

2. **Access Revocation:**
   - Disable all accounts on last day
   - Revoke all physical access badges
   - Collect company equipment
   - Remove from distribution lists

3. **Data Handoff:**
   - Manager identifies data owner
   - Transfer files and documents
   - Reassign permissions
   - Update documentation

4. **Post-Employment:**
   - Forward email for 30 days if required
   - Remove from company directory
   - Archive email and files

### 9.2 Involuntary Termination

**Timeline: Immediate**

**Process:**

1. **HR Notification:**
   - HR notifies IT and Security
   - Reason provided (if relevant to access)
   - Immediate revocation requested

2. **Immediate Access Revocation:**
   - Disable all accounts within 15 minutes
   - Revoke all credentials
   - Terminate active sessions
   - Collect equipment

3. **Security Assessment:**
   - Review recent access logs
   - Check for data exfiltration
   - Interview manager if needed
   - Preserve evidence if required

### 9.3 Contract Expiration

**Timeline:**
- One week before: Notify manager access will expire
- Last day: Revoke all access
- Manager can request extension with justification

**Process:**

1. **Notification:**
   - Automated email to manager and contractor
   - List of access that will expire
   - Extension request link

2. **Revocation:**
   - Automatic access expiration
   - Remove from systems
   - Archive work products

3. **Extension:**
   - Manager submits extension request
   - Includes justification and duration
   - Security approval
   - Access restored if approved

### 9.4 Access Modification

**Role Change:**

1. Manager submits access modification request
2. IT reviews new role requirements
3. Old access removed within 48 hours
4. New access granted within 24 hours
5. User notified of changes

**Project Completion:**

1. Automated notification to manager
2. Review of project-specific access
3. Remove unnecessary access
4. Document justification for retained access

---

## 10. Access Logging and Monitoring

### 10.1 Audit Events Logged

All access events are logged:
- Successful and failed login attempts
- Privileged access use
- Access to sensitive data
- Permission changes
- Account creation, modification, deletion

### 10.2 Log Retention

- Access logs: 90 days (hot storage)
- Audit logs: 1 year (cold storage)
- Archive logs: 7 years (SOC2 requirement)

### 10.3 Monitoring and Alerting

**Real-Time Alerts:**
- Multiple failed login attempts
- Access from unusual location
- Privileged access outside business hours
- Access to sensitive data by non-authorized users
- Permission escalations

**Review:**
- Daily: Security team reviews alerts
- Weekly: Access trend analysis
- Monthly: Anomaly investigation
- Quarterly: Full audit log review

---

## 11. Third-Party Access

### 11.1 Contractor Access

**Requirements:**
- Signed contract with confidentiality provisions
- Background check (if appropriate)
- Manager sponsorship
- Limited to required systems
- Time-limited access
- Monitoring and audit logging

**Process:**
- Treated as temporary employee
- Manager responsible for access
- Quarterly review
- Automatic expiration

### 11.2 Vendor Access

**Requirements:**
- Contract with data processing provisions
- NDA signed
- Access limited to support function
- Session recording required
- No access to customer data without approval

**Process:**
- Just-in-time access where possible
- Time-limited sessions
- Active monitoring
- Immediate revocation upon completion

### 11.3 Customer Access

**Self-Service:**
- Customers access only their own data
- No access to other customers' data
- Audit logging of all access
- Session timeout: 30 minutes

**Support Access:**
- Customer support can view with customer permission
- All access logged with justification
- Customer notified of access

---

## 12. Access Control Automation

### 12.1 Automated Provisioning

**Integration:**
- HRIS → IT system for employee onboarding
- Automatic account creation
- Role-based access assignment
- Welcome email with credentials

**Tools:**
- Okta for identity management
- AWS IAM for AWS access
- GitHub for source control
- Slack for communication

### 12.2 Automated Deprovisioning

**Trigger:**
- HR status change
- Contract expiration
- Manager request

**Actions:**
- Disable all accounts
- Revoke all permissions
- Send notification to manager
- Log deprovisioning

### 12.3 Access Certification

**Automated Reminders:**
- Quarterly access review reminders
- Annual certification reminders
- Event-driven review triggers

**Workflow:**
- Generate access reports
- Send to managers for review
- Track responses
- Execute approved changes

---

## 13. Compliance and Evidence

### 13.1 Evidence Requirements

**For SOC2 Audit:**
- Access request forms and approvals
- Access review reports and certifications
- Audit logs of access changes
- Privileged access justification
- MFA enforcement evidence

### 13.2 Evidence Collection

**Automated Collection:**
- Access logs from all systems
- Permission change history
- Login/logout logs
- Privileged access logs

**Manual Collection:**
- Signed access request forms
- Access review certifications
- Exception approvals

### 13.3 Reporting

**Quarterly Reports:**
- Access provisioned
- Access revoked
- Access review results
- Privileged access trends

**Annual Reports:**
- Access certification results
- Access control program maturity
- Recommended improvements

---

## 14. Related Documents

- [SOC2 Security Policies](./policies.md)
- [SOC2 Incident Response Procedures](./incident_response.md)
- [SOC2 Change Management Procedures](./change_management.md)
- [SOC2 Data Classification Schema](./data_classification.md)
- [SOC2 Audit Log Requirements](./audit_log_requirements.md)
- [SOC2 Checklist](./soc2_checklist.md)

---

**END OF PROCEDURE DOCUMENT**
