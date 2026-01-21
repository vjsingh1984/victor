# GDPR Data Processing Documentation - Victor AI

**Version:** 1.0
**Last Updated:** 2026-01-20
**Next Review:** 2026-07-20
**Owner:** Data Protection Officer (DPO)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Legal Basis for Processing](#2-legal-basis-for-processing)
3. [Data Processing Activities](#3-data-processing-activities)
4. [Data Processor Relationships](#4-data-processor-relationships)
5. [Data Transfers](#5-data-transfers)
6. [Processing Records](#6-processing-records)

---

## 1. Overview

### 1.1 Purpose

This document describes Victor AI's data processing activities in accordance with GDPR Articles 30 (Records of Processing Activities).

### 1.2 Scope

This document covers:
- All personal data processing by Victor AI
- All data subjects (customers, employees, website visitors)
- All processing purposes
- All data recipients

### 1.3 Data Controller vs. Data Processor

**Victor AI as Data Controller:**
- Customer data collected directly from users
- Employee data
- Website visitor data
- Marketing data

**Victor AI as Data Processor:**
- When analyzing customer code on behalf of customers
- When providing AI coding assistant services

---

## 2. Legal Basis for Processing

### 2.1 Legal Bases Used

**Article 6(1) GDPR Legal Bases:**

| Data Type | Legal Basis | Justification |
|-----------|-------------|---------------|
| **Customer Account Data** | Contract (Art. 6(1)(b)) | Necessary to provide service under contract |
| **Customer Conversations** | Contract (Art. 6(1)(b)) | Necessary to provide AI assistant service |
| **Service Usage Data** | Legitimate Interest (Art. 6(1)(f)) | Service improvement, security, analytics |
| **Marketing Data** | Consent (Art. 6(1)(a)) | User consented to marketing communications |
| **Employee Data** | Legal Obligation (Art. 6(1)(c)) | Employment law, tax, social security |
| **Security Logs** | Legitimate Interest (Art. 6(1)(f)) | Security, fraud prevention |
| **Website Analytics** | Consent (Art. 6(1)(a)) | User consented to cookies |

### 2.2 Special Category Data

**Article 9 GDPR - Not Currently Applicable**

Victor AI does not currently process special category data (health, biometric, genetic, etc.). If this changes in the future, explicit consent or another Article 9(2) basis will be obtained.

### 2.3 Criminal Offense Data

**Article 10 GDPR - Not Currently Applicable**

Victor AI does not process data related to criminal convictions and offenses.

---

## 3. Data Processing Activities

### 3.1 Customer Data Processing

**Purpose:** Provide AI coding assistant service

**Data Categories:**
- Personal identifiers: name, email address, username
- Account credentials: password hash, MFA secrets
- Conversation history: user prompts, AI responses
- Usage data: features used, time spent, frequency
- Technical data: IP address, device information, browser

**Processing Operations:**
- Collection: Account registration, service usage
- Storage: Encrypted storage in databases
- Analysis: Code analysis, conversation processing
- Deletion: Upon account closure or retention expiration

**Data Subjects:** Customers (individual users)

**Recipients:**
- Internal: Engineering (for development), Support (for assistance)
- External: Cloud providers (AWS), AI model providers

**Transfers:**
- Within EEA: Stored in EU region
- Outside EEA: To US cloud providers (AWS) with SCCs

**Retention:**
- Account data: Duration of relationship + 90 days
- Conversation history: 90 days
- Usage analytics: 365 days

**Security Measures:**
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Access control (authentication, authorization)
- Logging and monitoring

### 3.2 Employee Data Processing

**Purpose:** Employment relationship management

**Data Categories:**
- Personal identifiers: name, address, SSN/Tax ID
- Employment data: role, salary, performance
- Contact data: email, phone, emergency contact
- Time and attendance: work hours, leave
- Benefits: health insurance, retirement

**Processing Operations:**
- Collection: During employment, benefits enrollment
- Storage: HRIS system, payroll system
- Processing: Payroll, benefits administration, performance management
- Deletion: 7 years after termination (legal requirement)

**Data Subjects:** Employees, contractors

**Recipients:**
- Internal: HR, Finance, Management
- External: Payroll provider, benefits providers, tax authorities

**Transfers:**
- Within EEA: Primary storage
- Outside EEA: To US payroll/benefits providers with SCCs

**Retention:**
- Active employment: Duration of employment
- Post-employment: 7 years (tax, legal requirement)

**Security Measures:**
- Access restricted to HR and authorized personnel
- Encryption at rest and in transit
- Audit logging
- Confidentiality agreements

### 3.3 Website Visitor Data

**Purpose:** Website functionality, analytics, marketing

**Data Categories:**
- Technical data: IP address, browser, device, referrer
- Behavioral data: Pages visited, time on page, clicks
- Cookie data: Analytics cookies, marketing cookies
- Form data: Contact form submissions, newsletter signup

**Processing Operations:**
- Collection: Cookies, form submissions
- Storage: Analytics platforms, CRM
- Analysis: User behavior analysis, conversion tracking
- Deletion: Based on cookie consent and retention policy

**Data Subjects:** Website visitors

**Recipients:**
- Internal: Marketing, Product
- External: Analytics providers (Google Analytics), CRM

**Transfers:**
- Within EEA: Primary analytics
- Outside EEA: To US analytics providers with consent

**Retention:**
- Analytics data: 26 months (Google Analytics default)
- Form submissions: Until request processed or consent withdrawn
- Cookie data: Based on cookie category (session to 1 year)

**Security Measures:**
- HTTPS for data transmission
- Cookie consent management
- Data anonymization (IP masking)
- Access controls

---

## 4. Data Processor Relationships

### 4.1 Data Processors

**Definition:** Data processors process personal data on behalf of Victor AI (the controller).

**Key Data Processors:**

| Processor | Processing Activity | Location | SCC Signed? |
|-----------|---------------------|----------|-------------|
| **AWS** | Cloud infrastructure, data storage | EU, US | Yes |
| **OpenAI** | AI model inference | US | Yes |
| **Anthropic** | AI model inference | US | Yes |
| **Google Cloud** | Analytics | US | Yes |
| **Stripe** | Payment processing | EU, US | Yes |
| **GitHub** | Source code hosting | US | Yes |

### 4.2 Data Processing Agreements (DPAs)

**Required Clauses (GDPR Article 28):**

1. **Processor acts only on controller's documented instructions**
2. **Processor ensures persons processing data have confidentiality commitments**
3. **Processor implements appropriate technical and organizational measures**
4. **Processor assists controller with data subject rights**
5. **Processor assists controller with security obligations**
6. **Processor assists controller with breach notification**
7. **Processor makes available to controller all information for compliance**
8. **Processor deletes or returns all personal data after services end**
9. **Processor allows for audits and inspections**
10. **Processor may only subcontract with controller's prior authorization**

### 4.3 Subprocessor Management

**Process:**

1. **Identify Subprocessor:**
   - Vendor identifies need for subprocessor
   - Vendor submits subprocessor details to Victor AI

2. **Assessment:**
   - Security assessment
   - Data processing review
   - GDPR compliance verification

3. **Approval:**
   - Victor AI approves or rejects subprocessor
   - Update DPA to include subprocessor

4. **Ongoing Monitoring:**
   - Monitor subprocessor compliance
   - Review subprocessor performance

---

## 5. Data Transfers

### 5.1 Transfers Outside EEA

**Mechanisms:**

1. **Standard Contractual Clauses (SCCs):**
   - Adopted: European Commission SCCs (2021 edition)
   - Signed with: All non-EEA processors
   - Version: Module 1 (Controller to Processor), Module 2 (Processor to Subprocessor), Module 3 (Controller to Controller)

2. **Adequacy Decisions:**
   - Not currently relied upon (US adequacy uncertain post-Schrems II)

3. **Binding Corporate Rules (BCRs):**
   - Not applicable (Victor AI is not part of a corporate group)

### 5.2 Transfer Impact Assessment (TIA)

**Required when:**
- Relying on SCCs
- Transfer to countries without adequacy decision
- High-risk transfer (e.g., large-scale data, sensitive data)

**TIA Process:**

1. **Describe Transfer:**
   - What data is being transferred
   - To which country
   - For what purposes
   - How often

2. **Assess Laws and Practices:**
   - Data protection laws in destination country
   - Government access powers
   - Surveillance practices
   - Data localization requirements

3. **Implement Supplementary Measures:**
   - Technical measures (encryption, pseudonymization)
   - Contractual measures (SCCs, DPAs)
   - Organizational measures (policies, training)

4. **Document and Review:**
   - Document TIA
   - Review annually
   - Update if circumstances change

**Example TIA - AWS Transfer:**

```yaml
transfer_impact_assessment:
  transfer_id: "TIA-2026-001"
  date: "2026-01-20"
  reviewer: "DPO"

  transfer_details:
    data_exporter: "Victor AI (EEA)"
    data_importer: "AWS (US)"
    data_categories:
      - "Customer account data"
      - "Customer conversation history"
      - "Usage analytics"
    volume: "Approximately 10,000 customer records"
    frequency: "Continuous (real-time synchronization)"
    purposes: "Service provision, data storage, analytics"

  legal_framework:
    destination_country: "United States"
    adequacy_decision: "No"
    sccs_signed: "Yes (Module 1)"
    schrems_ii_compliant: "Yes (implemented supplementary measures)"

  risk_assessment:
    government_access_risk: "Medium"
    surveillance_concerns: "Yes (Cloud Act, FISA 702)"
    data_protection_level: "Strong (AWS implements GDPR-level protections)"

  supplementary_measures:
    technical:
      - "Encryption at rest (AES-256)"
      - "Encryption in transit (TLS 1.3)"
      - "Customer-managed encryption keys (option)"
    organizational:
      - "AWS GDPR DPA"
      - "AWS security certifications (SOC2, ISO 27001)"
      - "Regular security assessments"

  conclusion: "Transfer can proceed with SCCs and supplementary measures"
  review_date: "2027-01-20"
```

### 5.3 International Data Transfer Policy

**Policy:**
- Prefer EEA-based processors when feasible
- Always use SCCs for non-EEA transfers
- Conduct TIA for high-risk transfers
- Implement supplementary measures as needed
- Monitor legal developments (e.g., new adequacy decisions)
- Review transfers annually

---

## 6. Processing Records

### 6.1 Record of Processing Activities (ROPA)

**Required by:** GDPR Article 30

**Format:** Maintain written record (electronic or paper)

**Content:**

```yaml
record_of_processing_activities:
  controller: "Victor AI Ltd"
  representative: "[Name]"
  dpo: "dpo@victor.ai"

  processing_activities:
    - activity_id: "PA-001"
      activity_name: "Customer Service Provision"
      purposes:
        - "Provide AI coding assistant service"
        - "Customer support"
        - "Service improvement"
      categories_data_subjects:
        - "Customers (individual users)"
      categories_personal_data:
        - "Identification: name, email, username"
        - "Account: password hash, MFA secrets"
        - "Conversations: prompts, responses"
        - "Usage: features used, time spent"
      categories_recipients:
        - "Internal: Engineering, Support"
        - "External: AWS, OpenAI, Anthropic"
      transfers_to_third_countries:
        - destination: "United States"
          safeguards: "SCCs + supplementary measures"
          frequency: "Continuous"
      retention_periods:
        - "Account data: Duration of relationship + 90 days"
        - "Conversations: 90 days"
        - "Usage data: 365 days"
      security_measures:
        - "Encryption at rest (AES-256)"
        - "Encryption in transit (TLS 1.3)"
        - "Access control (RBAC)"
        - "Logging and monitoring"
      legal_basis:
        - "Contract (Article 6(1)(b))"
      date_created: "2026-01-01"
      last_updated: "2026-01-20"

    - activity_id: "PA-002"
      activity_name: "Employee Data Management"
      purposes:
        - "Employment relationship management"
        - "Payroll administration"
        - "Benefits administration"
      categories_data_subjects:
        - "Employees"
        - "Contractors"
      categories_personal_data:
        - "Identification: name, address, SSN/Tax ID"
        - "Employment: role, salary, performance"
        - "Contact: email, phone, emergency contact"
      categories_recipients:
        - "Internal: HR, Finance, Management"
        - "External: Payroll provider, benefits providers"
      transfers_to_third_countries:
        - destination: "United States"
          safeguards: "SCCs"
          frequency: "Monthly (payroll)"
      retention_periods:
        - "Active employment: Duration of employment"
        - "Post-employment: 7 years"
      security_measures:
        - "Encryption at rest and in transit"
        - "Access restricted to HR and authorized personnel"
        - "Audit logging"
        - "Confidentiality agreements"
      legal_basis:
        - "Legal obligation (Article 6(1)(c))"
        - "Contract (Article 6(1)(b))"
      date_created: "2026-01-01"
      last_updated: "2026-01-20"
```

### 6.2 ROPA Maintenance

**Update When:**
- New processing activity started
- Existing processing activity changed
- Processing activity stopped
- Data recipients changed
- Legal basis changed
- Retention periods changed
- Security measures updated

**Review:** Quarterly

**Access:**
- DPO: Full access
- Management: Read access
- Auditors: Read access upon request
- Data subjects: Relevant portions upon request

---

## 7. Data Processing by Third Parties

### 7.1 AI Model Providers

**OpenAI, Anthropic, etc.:**

**Role:** Data Processor

**Processing:**
- Customer code snippets sent for analysis
- AI responses generated
- No training on customer data (contractually prohibited)

**Legal Basis:** Contract (Article 6(1)(b))

**SCCs:** Signed

**Data Location:** US

**Security Measures:**
- TLS for data transmission
- No data retention (immediate processing)
- No data used for training (contractual guarantee)

### 7.2 Cloud Infrastructure Providers

**AWS, GCP:**

**Role:** Data Processor

**Processing:**
- Data storage
- Compute services
- Networking

**Legal Basis:** Contract (Article 6(1)(b))

**SCCs:** Signed

**Data Location:** EU and US (multi-region)

**Security Measures:**
- SOC2 Type II certified
- ISO 27001 certified
- Customer-managed encryption keys available
- Regular penetration testing

---

## 8. Automated Decision Making

**Article 22 GDPR - Not Currently Applicable**

Victor AI does not currently engage in solely automated decision-making that produces legal or similarly significant effects.

**Future Consideration:**
If automated decision-making is implemented:
- Provide notice to data subjects
- Implement right to human intervention
- Provide right to express point of view
- Provide right to contest decision

---

## 9. Data Protection Impact Assessment (DPIA)

**Article 35 GDPR - When Required:**

A DPIA is required when processing is likely to result in high risk to data subjects, including:

- Systematic and extensive evaluation of personal aspects (profiling)
- Large-scale processing of special category data
- Large-scale systematic monitoring of public areas
- Processing using new technologies

**Current Status:**

Victor AI's current processing activities are not considered high-risk. However, a DPIA will be conducted if:

1. Large-scale processing of conversation data for AI training
2. Implementation of behavioral profiling
3. Integration of biometric authentication
4. Other high-risk processing

**DPIA Template Available:** See [Data Protection Impact Assessment](./data_protection_impact.md)

---

## 10. Related Documents

- [GDPR Data Subject Rights](./data_subject_rights.md)
- [GDPR Consent Management](./consent_management.md)
- [GDPR Data Breach Procedures](./data_breach_procedures.md)
- [GDPR Privacy Policy](./privacy_policy.md)
- [GDPR Data Protection Impact Assessment](./data_protection_impact.md)
- [GDPR Checklist](./gdpr_checklist.md)

---

**END OF DOCUMENT**
