# SOC2 Vendor Management Procedures - Victor AI

> **Template**: This document describes intended controls. It does not assert current certification or compliance. Update with actual audit evidence and operational details.


**Version:** 1.0
**Reading Time:** 8 min
**Last Updated:** 2026-01-20
**Next Review:** 2026-04-20
**Owner:** Chief Information Security Officer (CISO)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Vendor Classification](#2-vendor-classification)
3. [Vendor Onboarding](#3-vendor-onboarding)
4. [Vendor Monitoring](#4-vendor-monitoring)
5. [Vendor Offboarding](#5-vendor-offboarding)
6. [Data Processing Agreements](#6-data-processing-agreements)
7. [Vendor Risk Assessment](#7-vendor-risk-assessment)

---

## 1. Overview

### 1.1 Purpose

This document defines the procedures for managing third-party vendors in accordance with SOC2 Trust Services Criteria
  (CC9.2).

### 1.2 Scope

This procedure applies to:
- All SaaS vendors
- All infrastructure providers (cloud, hosting)
- All professional services (consulting, development)
- All data processors

### 1.3 Objectives

- Ensure vendors meet security standards
- Protect customer data shared with vendors
- Monitor vendor compliance
- Manage vendor risk

---

## 2. Vendor Classification

### 2.1 Risk Categories

| Category | Criteria | Examples | Assessment Required |
|----------|----------|----------|---------------------|
| **Critical** | Access to customer data, system access, critical operations | - Cloud providers (AWS, GCP)<br>- Payment processors<br>- Email providers | Full security assessment, annual review, SOC2/ISO certification required |
| **High** | Access to confidential data, business impact | - CRM systems<br>- Development tools<br>- Project management | Security questionnaire, annual review |
| **Medium** | Access to internal data, some business impact | - Communication tools<br>- Design tools<br>- Analytics platforms | Basic questionnaire, biennial review |
| **Low** | No data access, minimal business impact | - Office supplies<br>- Basic utilities | No assessment required |

### 2.2 Vendor Inventory

Maintain current vendor inventory including:
- Vendor name and contact
- Services provided
- Data access (if any)
- Risk classification
- Contract dates
- Assessment status
- Certification status (SOC2, ISO 27001)

---

## 3. Vendor Onboarding

### 3.1 Onboarding Process

**For Critical/High Vendors:**

1. **Business Justification:**
   - Document need for vendor
   - Identify alternatives considered
   - Assess business value

2. **Security Assessment:**
   - Send security questionnaire
   - Request security documentation
   - Verify certifications (SOC2, ISO 27001)
   - Assess security controls

3. **Data Processing Review:**
   - Identify data to be shared
   - Assess data handling practices
   - Review data location (data residency)
   - Verify data protection measures

4. **Legal Review:**
   - Review terms of service
   - Create Data Processing Agreement (DPA)
   - Review liability and indemnification
   - Verify breach notification requirements

5. **Contract Negotiation:**
   - Include security requirements
   - Include right to audit
   - Include breach notification SLA
   - Include data deletion upon termination

6. **Approval:**
   - Business owner approval
   - Legal approval
   - Security approval
   - Executive approval for critical vendors

7. **Onboarding:**
   - Sign contracts
   - Configure access controls
   - Implement monitoring
   - Add to vendor inventory

**For Medium/Low Vendors:**

1. Complete basic questionnaire
2. Review terms of service
3. Approve if acceptable
4. Add to vendor inventory

### 3.2 Security Questionnaire

**Critical/High Vendors - Required Information:**

```yaml
vendor_security_assessment:
  vendor_name: "Vendor Name"
  assessment_date: "2026-01-20"

  company_information:
    business_address: "Address"
    data_centers_locations: ["List locations"]
    years_in_business: 10

  certifications:
    soc2_type_ii: true
    soc2_report_date: "2025-06-01"
    iso_27001: true
    iso_27001_certificate_date: "2025-03-01"
    other_certifications: ["List other certifications"]

  security_practices:
    encryption_in_transit: true
    encryption_at_rest: true
    encryption_standards: "TLS 1.3, AES-256"
    access_controls: true
    mfa_required: true
    annual_penetration_testing: true
    vulnerability_scanning: true
    incident_response_plan: true
    security_training: true

  data_protection:
    data_classification: true
    dpo_appointed: true
    gdpr_compliant: true
    data_breach_notification: "Within 24 hours"
    data_deletion_policy: true

  compliance:
    privacy_policy: "URL"
    subprocessors: "List of subprocessors"
    data_locations: ["Countries where data stored"]
    data_export_restrictions: "Any restrictions"

  incident_response:
    security_incidents_last_12_months: 0
    data_breaches_last_12_months: 0
    breach_notification_process: "Description"
    24_7_security_contact: "security@vendor.com"

  business_continuity:
    sla_uptime: "99.9%"
    disaster_recovery_plan: true
    backup_frequency: "Daily"
    recovery_time_objective: "4 hours"
    recovery_point_objective: "1 hour"
```text

---

## 4. Vendor Monitoring

### 4.1 Ongoing Monitoring

**Critical Vendors:**
- Quarterly security check
- Annual reassessment
- Monitor security bulletins
- Review audit reports (if available)
- Assess any security incidents
- Verify certification currency

**High Vendors:**
- Annual security check
- Monitor security bulletins
- Review any security incidents

**Medium Vendors:**
- Biennial review
- Monitor for major issues

**Low Vendors:**
- No formal monitoring (as needed)

### 4.2 Monitoring Activities

**Automated Monitoring:**
- Subscribe to vendor security newsletters
- Monitor for security announcements
- Track certification expiration
- Monitor for news about vendor

**Manual Reviews:**
- Annual questionnaires for critical vendors
- Contract renewal reviews
- Performance reviews

### 4.3 Vendor Incidents

**If Vendor Reports Security Incident:**

1. **Assess Impact:**
   - Determine if Victor AI data affected
   - Assess potential customer impact
   - Identify affected systems/data

2. **Containment:**
   - Rotate credentials if vendor credentials compromised
   - Monitor for suspicious activity
   - Restrict vendor access if needed

3. **Communication:**
   - Internal notification to security team
   - Customer notification if data affected
   - Regulatory notification if required

4. **Remediation:**
   - Work with vendor on remediation
   - Review vendor relationship
   - Consider alternative vendors if severe

5. **Documentation:**
   - Document incident
   - Update vendor risk assessment
   - Review and improve vendor selection process

---

## 5. Vendor Offboarding

### 5.1 Offboarding Triggers

- Contract expiration not renewed
- Contract termination (cause or convenience)
- Security incident
- Vendor acquisition by another company
- Business no longer requires service

### 5.2 Offboarding Process

**1. Planning (30 days before expiration):**
- Review contract terms
- Identify data with vendor
- Plan migration strategy
- Identify replacement vendor (if needed)

**2. Data Migration/Export:**
- Export all Victor AI data
- Verify data completeness
- Validate data integrity
- Import to new system or archive

**3. Access Revocation:**
- Disable all vendor accounts
- Revoke API keys
- Remove vendor from systems
- Update firewall rules

**4. Data Deletion:**
- Request data deletion from vendor
- Obtain certificate of deletion
- Verify deletion (if critical)

**5. Documentation:**
- Document offboarding process
- Update vendor inventory
- Archive contract and assessments
- Document lessons learned

**6. Notification:**
- Notify teams of vendor change
- Update documentation
- Train on new vendor (if applicable)

---

## 6. Data Processing Agreements

### 6.1 Required Clauses

All vendors processing customer data must have DPA including:

**Data Protection:**
- Data classification requirements
- Encryption requirements
- Access control requirements
- Data retention limits
- Data deletion upon termination

**Security:**
- Security standards required
- Right to audit
- Breach notification requirements (24 hours)
- Security incident response

**Compliance:**
- GDPR compliance requirements
- SOC2 support requirements
- Data residency requirements
- Subprocessor restrictions (require notification)

**Liability:**
- Indemnification for data breaches
- Limitation of liability
- Insurance requirements

### 6.2 DPA Template

```markdown
# Data Processing Agreement - Victor AI and [Vendor]

**Effective Date:** [Date]
**Version:** 1.0

## 1. Data Processing

**Scope of Processing:**
[Description of services and data processed]

**Data Types:**
- Customer PII: [List data types]
- Usage Data: [List data types]
- Other Data: [List data types]

**Data Categories:**
- Classification: [CONFIDENTIAL/RESTRICTED]
- Special Category Data: [If applicable]

## 2. Security Measures

**Technical Measures:**
- Encryption at rest: AES-256
- Encryption in transit: TLS 1.3
- Access controls: Role-based, MFA
- Authentication: [Requirements]

**Organizational Measures:**
- Security training: Annual
- Background checks: For relevant staff
- Incident response: 24/7 capability
- Penetration testing: Annual

## 3. Data Subject Rights

**Support:**
- Right to access: [Process]
- Right to rectification: [Process]
- Right to erasure: [Process]
- Right to portability: [Process]
- Right to object: [Process]

**Response Time:** Within 30 days

## 4. Breach Notification

**Notification Timeline:** Within 24 hours of discovery
**Notification Content:**
- Nature of breach
- Data categories affected
- Individuals affected
- Mitigation measures
- Contact information

## 5. Audit Rights

**Frequency:** Annually or upon incident
**Scope:** Security controls, data processing
**Notice:** 30 days
**Cost:** Victor AI bears cost (unless finding non-compliance)

## 6. Subprocessors

**Restrictions:** Prior written consent required
**Approval Process:** Security assessment required
**Right to Object:** Victor AI may object on security grounds

## 7. Data Deletion

**Upon Termination:** Vendor must delete all Victor AI data within 30 days
**Verification:** Certificate of deletion provided

## 8. Term and Termination

**Term:** Same as master agreement
**Survival:** Data processing obligations survive termination

**Signatures:**
Victor AI: _________________ Date: ________
Vendor: _________________ Date: ________
```

---

## 7. Vendor Risk Assessment

### 7.1 Risk Assessment Process

**1. Identify Risks:**
- Security risks (data breach, unauthorized access)
- Operational risks (service disruption, poor performance)
- Compliance risks (regulatory violations)
- Financial risks (unexpected costs, penalties)
- Reputational risks (association with vendor)

**2. Assess Likelihood and Impact:**

| Risk | Likelihood | Impact | Risk Level | Mitigation |
|------|------------|--------|------------|------------|
| Data breach | Low | High | Medium | Encryption, audit |
| Service disruption | Medium | High | High | SLA, backup |
| Non-compliance | Low | High | Medium | Contractual terms |
| Price increase | High | Low | Low | Contract terms |

**3. Determine Risk Acceptance:**
- **Critical vendors:** High risk acceptable only with senior approval and mitigation
- **High vendors:** Medium risk acceptable with mitigation
- **Medium/Low vendors:** Low-medium risk acceptable

**4. Implement Mitigation:**
- Contractual protections
- Technical controls
- Monitoring
- Exit strategy

### 7.2 Risk Review

**Quarterly:**
- Review critical vendor risks
- Assess any new risks
- Update mitigation measures

**Annually:**
- Full risk reassessment
- Update vendor inventory
- Review vendor relationships

---

## 8. Approved Vendor List

### 8.1 Critical Vendors (2026)

| Vendor | Service | Data Access | Certifications | Risk Level | Next Review |
|--------|---------|-------------|----------------|------------|-------------|
| AWS | Cloud infrastructure | Yes | SOC2, ISO 27001 | Medium | 2026-07-01 |
| GitHub | Source control | Yes (code) | SOC2 | Medium | 2026-07-01 |
| [Add more] | | | | | |

### 8.2 High Vendors (2026)

| Vendor | Service | Data Access | Certifications | Risk Level | Next Review |
|--------|---------|-------------|----------------|------------|-------------|
| [Add vendors] | | | | | |

---

## 9. Related Documents

- [SOC2 Security Policies](./policies.md)
- [SOC2 Access Control Procedures](./access_control.md)
- [SOC2 Incident Response Procedures](./incident_response.md)
- [SOC2 Change Management Procedures](./change_management.md)
- [SOC2 Data Classification Schema](./data_classification.md)
- [SOC2 Checklist](./soc2_checklist.md)

---

**END OF PROCEDURE DOCUMENT**
