# SOC2 Access Control Procedures - Part 2

**Part 2 of 2:** Access Logging and Monitoring, Third-Party Access, Access Control Automation, and Compliance

---

## Navigation

- [Part 1: Access Management](part-1-access-management.md)
- **[Part 2: Logging, Third-Party, Automation, Compliance](#)** (Current)
- [**Complete Guide](../access_control.md)**

---

> **Template**: This document describes intended controls. It does not assert current certification or compliance. Update with actual audit evidence and operational details.

---

## 10. Access Logging and Monitoring

### 10.1 Access Logs

**Required Logging:**
- All access attempts (success and failure)
- Privileged access activities
- Access changes (grants, revocations)
- Authentication events
- Authorization failures

**Log Contents:**
- Timestamp
- User identity
- Resource accessed
- Action performed
- Result (success/failure)
- Source IP address

**Retention:** 90 days

### 10.2 Monitoring

**Real-time Monitoring:**
- Failed authentication attempts
- Unusual access patterns
- Privileged access usage
- Off-hours access

**Alert Thresholds:**
- 5 failed logins: Alert security team
- 10 failed logins: Lock account
- Privileged access outside business hours: Alert manager

---

## 11. Third-Party Access

### 11.1 Third-Party Requirements

**Allowed Third Parties:**
- Vendors with contractual relationship
- Consultants with NDA
- Auditors with engagement letter

**Approval Process:**
1. Business justification
2. Risk assessment
3. Manager approval
4. Limited time access
5. Monitoring requirements

### 11.2 Third-Party Controls

**Required Controls:**
- Non-disclosure agreement (NDA)
- Limited access scope
- Activity monitoring
- Time-bound access
- No data exfiltration

---

## 12. Access Control Automation

### 12.1 Automated Provisioning

**Use Automated Systems for:**
- User account creation
- Access grants based on role
- Default permissions assignment

### 12.2 Automated Deprovisioning

**Triggers:**
- Employee termination
- Contract end date
- Role change
- Access review failure

**Process:**
1. Disable account immediately
2. Revoke all access
3. Remove from groups
4. Notify manager

---

## 13. Compliance and Evidence

### 13.1 Required Evidence

**Access Records:**
- User access requests
- Approval documentation
- Access grants and revocations
- Access review results
- Exception authorizations

**Retention:** 7 years

### 13.2 SOC2 Compliance Mapping

**Control Requirements:**
- CC6.1: Logical and physical access controls
- CC6.7: Prevent access by unauthorized persons
- CC6.8: Access restriction based on roles

**Evidence Collection:**
- Automated access logs
- Review documentation
- Approval records
- Exception logs

---

## 14. Related Documents

- [Incident Response](./incident_response/)
- [Change Management](./change_management/)
- [Security Policy](../../../policies/security/)

---

**Version:** 1.0
**Reading Time:** 2 min
**Last Updated:** 2026-01-20
**Next Review:** 2026-04-20
**Owner:** DevOps Lead
