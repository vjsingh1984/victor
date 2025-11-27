# Victor Multi-User Authentication System
## Requirements & Design Specification

**Version**: 1.0
**Date**: 2025-11-26
**Status**: 📋 Design Phase
**Target Completion**: 2-3 weeks

---

## Table of Contents

1. [Requirements Analysis](#requirements-analysis)
2. [System Architecture](#system-architecture)
3. [Database Design](#database-design)
4. [Security Specifications](#security-specifications)
5. [API Endpoints](#api-endpoints)
6. [Frontend Components](#frontend-components)
7. [Implementation Phases](#implementation-phases)
8. [Testing Strategy](#testing-strategy)

---

## 1. Requirements Analysis

### 1.1 Functional Requirements

#### FR-1: User Registration (Self-Signup)
**Description**: New users can create accounts through a self-service signup form.

**Acceptance Criteria**:
- [ ] User provides: username, email, password, password confirmation
- [ ] Password must be entered twice and match
- [ ] Username must be unique (case-insensitive)
- [ ] Email must be unique and valid format
- [ ] CAPTCHA verification required to prevent bots
- [ ] Password strength requirements enforced (see Security)
- [ ] Account created in "active" state immediately
- [ ] User automatically logged in after signup
- [ ] Confirmation email sent (optional, Phase 2)

**User Story**:
> As a new user, I want to create an account by providing a username, email, and password, so that I can access the Victor AI assistant with my own personalized sessions.

---

#### FR-2: User Login
**Description**: Registered users can authenticate with username/password.

**Acceptance Criteria**:
- [ ] User provides username and password
- [ ] Credentials validated against encrypted password store
- [ ] JWT token issued on successful authentication
- [ ] Token includes: user_id, username, role, expiration
- [ ] Failed login attempts logged
- [ ] Account locked after 5 failed attempts (security)
- [ ] Session created and linked to user
- [ ] User redirected to chat interface

**User Story**:
> As a registered user, I want to log in with my username and password, so that I can access my private chat sessions and conversation history.

---

#### FR-3: Default Admin Account
**Description**: System ships with a default administrator account.

**Acceptance Criteria**:
- [ ] Default credentials: `admin` / `admin`
- [ ] Admin account created on first startup (database migration)
- [ ] Admin role assigned (elevated permissions)
- [ ] User prompted to change password on first login
- [ ] Warning displayed if default password still in use

**User Story**:
> As a system administrator, I want a default admin account with well-known credentials, so that I can access the system immediately after deployment and then change the password.

---

#### FR-4: Password Change
**Description**: Users can change their passwords.

**Acceptance Criteria**:
- [ ] User provides: current password, new password, new password confirmation
- [ ] Current password verified before change
- [ ] New password must meet strength requirements
- [ ] New password must differ from current password
- [ ] Password history maintained (prevent reuse of last 3 passwords)
- [ ] All active sessions invalidated after password change
- [ ] User notified via email (optional, Phase 2)

**User Story**:
> As a user, I want to change my password, so that I can maintain security and comply with password policies.

---

#### FR-5: Session Management
**Description**: User sessions are tracked and managed securely.

**Acceptance Criteria**:
- [ ] Each user has isolated chat sessions
- [ ] Sessions linked to user account (user_id foreign key)
- [ ] Users can only access their own sessions
- [ ] Session list shows only user's sessions
- [ ] Admin can view all sessions (audit capability)
- [ ] Sessions persist across logins
- [ ] Sessions cleaned up on account deletion

**User Story**:
> As a user, I want my chat sessions to be private and persistent, so that only I can access my conversation history.

---

#### FR-6: CAPTCHA Protection
**Description**: Bot prevention on signup and login (after N failures).

**Acceptance Criteria**:
- [ ] CAPTCHA displayed on signup form (always)
- [ ] CAPTCHA displayed on login after 3 failed attempts
- [ ] Integration with reCAPTCHA v3 or hCaptcha
- [ ] Backend validates CAPTCHA token
- [ ] Configurable CAPTCHA provider (environment variable)
- [ ] Graceful fallback if CAPTCHA service unavailable

**User Story**:
> As a system operator, I want CAPTCHA protection on signup and login, so that automated bots cannot abuse the system or perform credential stuffing attacks.

---

### 1.2 Non-Functional Requirements

#### NFR-1: Security
- Passwords encrypted with **bcrypt** (industry standard, cost factor 12)
- JWT tokens signed with **HS256** or **RS256** (configurable)
- Secrets stored in environment variables (never in code)
- HTTPS enforced in production (TLS 1.2+)
- SQL injection prevention (parameterized queries)
- XSS prevention (input sanitization, CSP headers)
- CSRF protection (SameSite cookies, CSRF tokens)

#### NFR-2: Performance
- Login response time: < 500ms
- Signup response time: < 1s (including CAPTCHA validation)
- Password hash verification: < 100ms
- Supports 1000+ concurrent authenticated users

#### NFR-3: Scalability
- Stateless authentication (JWT, no server sessions)
- Horizontal scaling supported (no session affinity)
- Database connection pooling (10-50 connections)

#### NFR-4: Usability
- Clear error messages (without exposing security details)
- Password strength indicator on forms
- "Forgot password" flow (Phase 2)
- Remember me option (Phase 2)

#### NFR-5: Compliance
- GDPR-ready (user data export/deletion)
- Password policy configurable
- Audit logging for authentication events

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (React + TypeScript)                              │
│  - LoginForm.tsx                                            │
│  - SignupForm.tsx                                           │
│  - PasswordChangeForm.tsx                                   │
│  - AuthContext (JWT storage, auth state)                    │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTPS + JWT
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Backend (FastAPI)                                          │
│  - /auth/signup          (POST)                             │
│  - /auth/login           (POST)                             │
│  - /auth/logout          (POST)                             │
│  - /auth/change-password (POST)                             │
│  - /auth/me              (GET)                              │
│  - Middleware: JWT verification                             │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────────┐
│Password │ │  JWT    │ │  CAPTCHA    │
│ Store   │ │ Service │ │  Service    │
│(bcrypt) │ │(PyJWT)  │ │(reCAPTCHA)  │
└─────────┘ └─────────┘ └─────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Database (SQLite / PostgreSQL)                             │
│  - users table                                              │
│  - sessions table (with user_id FK)                         │
│  - password_history table                                   │
│  - audit_log table                                          │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.2 Authentication Flow

#### 2.2.1 Signup Flow
```
User                Frontend            Backend             Database        CAPTCHA
  │                    │                   │                   │               │
  ├─Fill Form─────────►│                   │                   │               │
  │                    │                   │                   │               │
  ├─Submit────────────►│                   │                   │               │
  │                    ├─Verify CAPTCHA───────────────────────────────────────►│
  │                    │◄─Valid Token──────────────────────────────────────────┤
  │                    │                   │                   │               │
  │                    ├─POST /auth/signup►│                   │               │
  │                    │                   ├─Validate Input────►│               │
  │                    │                   ├─Hash Password     │               │
  │                    │                   ├─Create User───────►│               │
  │                    │                   │◄─User Created─────┤               │
  │                    │                   ├─Generate JWT      │               │
  │                    │◄─JWT Token────────┤                   │               │
  │◄─Success + Token───┤                   │                   │               │
  │                    │                   │                   │               │
  └─Redirect to Chat───┘                   │                   │               │
```

#### 2.2.2 Login Flow
```
User                Frontend            Backend             Database
  │                    │                   │                   │
  ├─Enter Credentials►│                   │                   │
  │                    │                   │                   │
  ├─Submit────────────►│                   │                   │
  │                    ├─POST /auth/login─►│                   │
  │                    │                   ├─Fetch User────────►│
  │                    │                   │◄─User Record──────┤
  │                    │                   ├─Verify Password   │
  │                    │                   ├─Generate JWT      │
  │                    │◄─JWT Token────────┤                   │
  │◄─Success + Token───┤                   │                   │
  │                    │                   │                   │
  ├─Store Token───────►│                   │                   │
  │(localStorage)      │                   │                   │
  │                    │                   │                   │
  └─Redirect to Chat───┘                   │                   │
```

#### 2.2.3 Authenticated Request Flow
```
User                Frontend            Backend (Middleware)   Handler
  │                    │                   │                      │
  ├─Action────────────►│                   │                      │
  │                    ├─GET /ws?token=JWT►│                      │
  │                    │                   ├─Verify JWT            │
  │                    │                   ├─Extract user_id       │
  │                    │                   ├─Add to request context│
  │                    │                   ├──────────────────────►│
  │                    │                   │                      ├─Process
  │                    │                   │                      ├─Access user data
  │                    │◄──────────────────┴──────────────────────┤
  │◄─Response──────────┤                                          │
```

---

## 3. Database Design

### 3.1 Schema

#### 3.1.1 Users Table
```sql
CREATE TABLE users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,  -- SQLite
    -- id           SERIAL PRIMARY KEY,                 -- PostgreSQL
    username        VARCHAR(50) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,              -- bcrypt hash
    role            VARCHAR(20) DEFAULT 'user',         -- 'admin' or 'user'
    is_active       BOOLEAN DEFAULT TRUE,
    is_locked       BOOLEAN DEFAULT FALSE,
    failed_login_attempts INTEGER DEFAULT 0,
    last_login      TIMESTAMP,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (username = LOWER(username)),                 -- Force lowercase
    CHECK (LENGTH(username) >= 3 AND LENGTH(username) <= 50),
    CHECK (role IN ('admin', 'user'))
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
```

#### 3.1.2 Sessions Table (Enhanced)
```sql
CREATE TABLE sessions (
    id              TEXT PRIMARY KEY,                   -- UUID
    user_id         INTEGER NOT NULL,
    title           VARCHAR(255) DEFAULT 'New Session',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_updated_at ON sessions(updated_at);
```

#### 3.1.3 Messages Table (New)
```sql
CREATE TABLE messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    sender          VARCHAR(20) NOT NULL,               -- 'user' or 'assistant'
    content         TEXT NOT NULL,
    kind            VARCHAR(20) DEFAULT 'normal',       -- 'normal' or 'tool'
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    CHECK (sender IN ('user', 'assistant')),
    CHECK (kind IN ('normal', 'tool'))
);

CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
```

#### 3.1.4 Password History Table
```sql
CREATE TABLE password_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_password_history_user_id ON password_history(user_id);
```

#### 3.1.5 Audit Log Table
```sql
CREATE TABLE audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER,                            -- NULL if user not found
    username        VARCHAR(50),
    event_type      VARCHAR(50) NOT NULL,               -- 'login', 'signup', 'logout', etc.
    success         BOOLEAN NOT NULL,
    ip_address      VARCHAR(45),                        -- IPv4 or IPv6
    user_agent      VARCHAR(255),
    details         TEXT,                               -- JSON metadata
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    CHECK (event_type IN ('signup', 'login', 'logout', 'password_change', 'failed_login', 'account_locked'))
);

CREATE INDEX idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_created_at ON audit_log(created_at);
```

---

### 3.2 Database Initialization

#### 3.2.1 Migration Script
```python
# migrations/001_create_auth_tables.py

import bcrypt
from datetime import datetime

def upgrade(connection):
    """Create authentication tables and default admin user."""
    cursor = connection.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) DEFAULT 'user',
            is_active BOOLEAN DEFAULT TRUE,
            is_locked BOOLEAN DEFAULT FALSE,
            failed_login_attempts INTEGER DEFAULT 0,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CHECK (username = LOWER(username)),
            CHECK (LENGTH(username) >= 3 AND LENGTH(username) <= 50),
            CHECK (role IN ('admin', 'user'))
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

    # Create other tables (sessions, password_history, audit_log)
    # ... (see schema above)

    # Create default admin user
    admin_password = "admin"
    password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt(rounds=12))

    cursor.execute("""
        INSERT OR IGNORE INTO users (username, email, password_hash, role)
        VALUES ('admin', 'admin@localhost', ?, 'admin')
    """, (password_hash.decode('utf-8'),))

    connection.commit()

def downgrade(connection):
    """Drop authentication tables."""
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS audit_log")
    cursor.execute("DROP TABLE IF EXISTS password_history")
    cursor.execute("DROP TABLE IF EXISTS messages")
    cursor.execute("DROP TABLE IF EXISTS sessions")
    cursor.execute("DROP TABLE IF EXISTS users")
    connection.commit()
```

---

## 4. Security Specifications

### 4.1 Password Requirements

#### Strength Policy
- **Minimum length**: 8 characters
- **Maximum length**: 128 characters
- **Required characters**: At least 3 of the following:
  - Uppercase letter (A-Z)
  - Lowercase letter (a-z)
  - Digit (0-9)
  - Special character (!@#$%^&*()_+-=[]{}|;:,.<>?)
- **Prohibited**: Common passwords (check against top 10,000 list)
- **Prohibited**: Username in password
- **Prohibited**: Sequential characters (123, abc)
- **Prohibited**: Repeated characters (aaa, 111)

#### Password Hashing
```python
import bcrypt

# Cost factor: 12 (recommended for 2025)
# Takes ~250ms on modern CPU (good balance)
password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))

# Verification
is_valid = bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
```

**Why bcrypt?**
- Industry standard
- Automatic salting
- Adaptive (cost factor increases over time)
- Resistant to GPU/ASIC attacks
- Battle-tested (20+ years)

---

### 4.2 JWT Token Specification

#### Token Structure
```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_id",           // Subject (user ID)
    "username": "john_doe",     // Username
    "email": "john@example.com", // Email
    "role": "user",             // Role
    "iat": 1701000000,          // Issued at (Unix timestamp)
    "exp": 1701086400,          // Expiration (Unix timestamp)
    "jti": "unique-token-id"    // JWT ID (for blacklisting)
  },
  "signature": "..."
}
```

#### Configuration
```python
# Environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # MUST be set (256-bit random)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24  # Token valid for 24 hours
JWT_REFRESH_ENABLED = False  # Phase 2 feature
```

#### Token Validation
```python
from jose import jwt, JWTError

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

### 4.3 CAPTCHA Integration

#### Supported Providers
1. **reCAPTCHA v3** (recommended)
   - Invisible CAPTCHA
   - Risk-based scoring
   - Free tier: 1M requests/month

2. **hCaptcha** (privacy-focused alternative)
   - GDPR compliant
   - Paid service

#### Backend Validation (reCAPTCHA v3)
```python
import httpx

async def verify_recaptcha(token: str, action: str = "signup") -> bool:
    """Verify reCAPTCHA v3 token."""
    secret_key = os.getenv("RECAPTCHA_SECRET_KEY")
    url = "https://www.google.com/recaptcha/api/siteverify"

    async with httpx.AsyncClient() as client:
        response = await client.post(url, data={
            "secret": secret_key,
            "response": token
        })
        result = response.json()

        # Check success and score (0.0 = bot, 1.0 = human)
        if result.get("success") and result.get("score", 0) >= 0.5:
            return True

    return False
```

#### Frontend Integration (React)
```tsx
import ReCAPTCHA from "react-google-recaptcha";

const SignupForm = () => {
    const recaptchaRef = useRef<ReCAPTCHA>(null);

    const handleSubmit = async () => {
        const token = await recaptchaRef.current?.executeAsync();
        // Send token to backend
    };

    return (
        <form onSubmit={handleSubmit}>
            {/* Form fields */}
            <ReCAPTCHA
                ref={recaptchaRef}
                size="invisible"
                sitekey={process.env.REACT_APP_RECAPTCHA_SITE_KEY}
            />
        </form>
    );
};
```

---

### 4.4 Account Lockout Policy

#### Trigger Conditions
- **5 failed login attempts** within 15 minutes
- Lockout duration: **30 minutes** (configurable)
- Admin accounts exempt (but logged)

#### Implementation
```python
async def handle_failed_login(username: str, db):
    user = await db.get_user_by_username(username)
    if not user:
        return  # Don't reveal if user exists

    user.failed_login_attempts += 1

    if user.failed_login_attempts >= 5:
        user.is_locked = True
        user.locked_until = datetime.utcnow() + timedelta(minutes=30)

        # Log event
        await db.log_audit_event(
            user_id=user.id,
            event_type="account_locked",
            success=False,
            details=f"Account locked after {user.failed_login_attempts} failed attempts"
        )

    await db.update_user(user)
```

---

## 5. API Endpoints

### 5.1 POST /auth/signup

**Description**: Register a new user account.

**Request**:
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123!",
  "password_confirm": "SecurePass123!",
  "captcha_token": "03AGdBq25..."
}
```

**Response (Success - 201)**:
```json
{
  "success": true,
  "message": "Account created successfully",
  "user": {
    "id": 42,
    "username": "john_doe",
    "email": "john@example.com",
    "role": "user"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (Error - 400)**:
```json
{
  "success": false,
  "error": "validation_error",
  "details": {
    "username": ["Username already exists"],
    "password": ["Password must contain at least one uppercase letter"]
  }
}
```

**Response (Error - 403)**:
```json
{
  "success": false,
  "error": "captcha_failed",
  "message": "CAPTCHA verification failed. Please try again."
}
```

---

### 5.2 POST /auth/login

**Description**: Authenticate user and issue JWT token.

**Request**:
```json
{
  "username": "john_doe",
  "password": "SecurePass123!",
  "captcha_token": "03AGdBq25..."  // Optional, required after 3 failures
}
```

**Response (Success - 200)**:
```json
{
  "success": true,
  "message": "Login successful",
  "user": {
    "id": 42,
    "username": "john_doe",
    "email": "john@example.com",
    "role": "user",
    "last_login": "2025-11-26T12:34:56Z"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (Error - 401)**:
```json
{
  "success": false,
  "error": "invalid_credentials",
  "message": "Invalid username or password",
  "failed_attempts": 2,
  "captcha_required": false
}
```

**Response (Error - 403)**:
```json
{
  "success": false,
  "error": "account_locked",
  "message": "Account locked due to too many failed login attempts. Please try again in 25 minutes.",
  "locked_until": "2025-11-26T13:00:00Z"
}
```

---

### 5.3 POST /auth/logout

**Description**: Invalidate current session (Phase 2: add token to blacklist).

**Headers**:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (Success - 200)**:
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

---

### 5.4 POST /auth/change-password

**Description**: Change user password.

**Headers**:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Request**:
```json
{
  "current_password": "OldPass123!",
  "new_password": "NewSecurePass456!",
  "new_password_confirm": "NewSecurePass456!"
}
```

**Response (Success - 200)**:
```json
{
  "success": true,
  "message": "Password changed successfully. Please log in again."
}
```

**Response (Error - 400)**:
```json
{
  "success": false,
  "error": "validation_error",
  "details": {
    "current_password": ["Current password is incorrect"],
    "new_password": ["New password cannot be the same as any of your last 3 passwords"]
  }
}
```

---

### 5.5 GET /auth/me

**Description**: Get current user information.

**Headers**:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response (Success - 200)**:
```json
{
  "success": true,
  "user": {
    "id": 42,
    "username": "john_doe",
    "email": "john@example.com",
    "role": "user",
    "created_at": "2025-11-01T10:00:00Z",
    "last_login": "2025-11-26T12:34:56Z",
    "is_active": true
  }
}
```

---

### 5.6 WebSocket Authentication

**Modification to existing /ws endpoint**:

**Before**:
```
ws://localhost:8000/ws?session_id=<uuid>
```

**After**:
```
ws://localhost:8000/ws?session_id=<uuid>&token=<jwt>
```

**Validation**:
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Extract and verify JWT token
    token = websocket.query_params.get("token")
    if not token:
        await websocket.send_text("[error] Authentication required")
        await websocket.close(code=1008, reason="Unauthorized")
        return

    try:
        payload = verify_token(token)
        user_id = payload["sub"]
    except Exception:
        await websocket.send_text("[error] Invalid or expired token")
        await websocket.close(code=1008, reason="Unauthorized")
        return

    # Link session to user
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())

    # Verify user owns this session
    session = await db.get_session(session_id)
    if session and session.user_id != user_id:
        await websocket.send_text("[error] Access denied")
        await websocket.close(code=1008, reason="Forbidden")
        return

    # Create session if new
    if not session:
        await db.create_session(session_id, user_id)

    # Continue with existing logic...
```

---

## 6. Frontend Components

### 6.1 Component Structure

```
web/ui/src/
├── components/
│   ├── auth/
│   │   ├── LoginForm.tsx          # Login form
│   │   ├── SignupForm.tsx         # Signup form
│   │   ├── PasswordChangeForm.tsx # Change password
│   │   ├── ProtectedRoute.tsx     # Route guard
│   │   └── AuthGuard.tsx          # Component wrapper
│   ├── Message.tsx                # (existing)
│   ├── MessageInput.tsx           # (existing)
│   └── ThemeToggle.tsx            # (existing)
├── contexts/
│   └── AuthContext.tsx            # Auth state management
├── hooks/
│   ├── useAuth.ts                 # Auth hook
│   └── useWebSocket.ts            # WebSocket with auth
├── services/
│   ├── authService.ts             # API calls
│   └── captchaService.ts          # CAPTCHA handling
├── App.tsx                        # (enhanced with auth)
└── main.tsx                       # (existing)
```

---

### 6.2 Key Components

#### 6.2.1 AuthContext.tsx
```typescript
import React, { createContext, useState, useEffect, useCallback } from 'react';

interface User {
  id: number;
  username: string;
  email: string;
  role: 'admin' | 'user';
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string, captchaToken?: string) => Promise<void>;
  signup: (username: string, email: string, password: string, passwordConfirm: string, captchaToken: string) => Promise<void>;
  logout: () => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string, newPasswordConfirm: string) => Promise<void>;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('auth_token'));
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (token) {
      // Fetch user profile
      fetchUserProfile(token).then(setUser).catch(() => {
        setToken(null);
        localStorage.removeItem('auth_token');
      }).finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, [token]);

  const login = useCallback(async (username: string, password: string, captchaToken?: string) => {
    const response = await authService.login(username, password, captchaToken);
    setToken(response.token);
    setUser(response.user);
    localStorage.setItem('auth_token', response.token);
  }, []);

  const signup = useCallback(async (username: string, email: string, password: string, passwordConfirm: string, captchaToken: string) => {
    const response = await authService.signup(username, email, password, passwordConfirm, captchaToken);
    setToken(response.token);
    setUser(response.user);
    localStorage.setItem('auth_token', response.token);
  }, []);

  const logout = useCallback(async () => {
    if (token) {
      await authService.logout(token);
    }
    setToken(null);
    setUser(null);
    localStorage.removeItem('auth_token');
  }, [token]);

  const changePassword = useCallback(async (currentPassword: string, newPassword: string, newPasswordConfirm: string) => {
    if (!token) throw new Error('Not authenticated');
    await authService.changePassword(token, currentPassword, newPassword, newPasswordConfirm);
    // Force re-login after password change
    await logout();
  }, [token, logout]);

  return (
    <AuthContext.Provider value={{
      user,
      token,
      isAuthenticated: !!user,
      isLoading,
      login,
      signup,
      logout,
      changePassword
    }}>
      {children}
    </AuthContext.Provider>
  );
};
```

#### 6.2.2 LoginForm.tsx (Partial)
```typescript
import { useState, useRef } from 'react';
import ReCAPTCHA from 'react-google-recaptcha';
import { useAuth } from '../hooks/useAuth';

export const LoginForm = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [failedAttempts, setFailedAttempts] = useState(0);
  const recaptchaRef = useRef<ReCAPTCHA>(null);
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    try {
      let captchaToken: string | undefined;

      // CAPTCHA required after 3 failed attempts
      if (failedAttempts >= 3) {
        captchaToken = await recaptchaRef.current?.executeAsync() || undefined;
      }

      await login(username, password, captchaToken);
      // Redirect handled by auth context
    } catch (err: any) {
      setError(err.message || 'Login failed');
      setFailedAttempts(prev => prev + 1);

      if (err.captcha_required) {
        // Show CAPTCHA
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-md mx-auto bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}

      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Username</label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          required
        />
      </div>

      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          required
        />
      </div>

      {failedAttempts >= 3 && (
        <ReCAPTCHA
          ref={recaptchaRef}
          size="invisible"
          sitekey={import.meta.env.VITE_RECAPTCHA_SITE_KEY}
        />
      )}

      <button
        type="submit"
        className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
      >
        Log In
      </button>
    </form>
  );
};
```

---

## 7. Implementation Phases

### Phase 1: Core Authentication (Week 1-2)
- [ ] Database schema and migrations
- [ ] User model and password hashing (bcrypt)
- [ ] JWT token generation and validation
- [ ] Backend API endpoints (/auth/signup, /auth/login, /auth/logout)
- [ ] Basic frontend forms (login, signup)
- [ ] AuthContext and useAuth hook
- [ ] WebSocket authentication modification
- [ ] Default admin account creation
- [ ] Unit tests for auth logic

**Deliverable**: Users can signup, login, and access authenticated chat sessions.

---

### Phase 2: Security Hardening (Week 3)
- [ ] CAPTCHA integration (reCAPTCHA v3)
- [ ] Account lockout policy (5 failed attempts)
- [ ] Password change functionality
- [ ] Password history (prevent reuse)
- [ ] Audit logging (all auth events)
- [ ] Session isolation (users see only their sessions)
- [ ] Protected routes (frontend)
- [ ] HTTPS enforcement (production)
- [ ] Integration tests for security flows

**Deliverable**: Production-grade security measures in place.

---

### Phase 3: UX & Advanced Features (Week 4+)
- [ ] Password strength indicator
- [ ] "Forgot password" flow (email-based)
- [ ] Email verification on signup
- [ ] "Remember me" option (longer-lived tokens)
- [ ] User profile management
- [ ] Admin dashboard (view all users, sessions)
- [ ] Two-factor authentication (2FA) (optional)
- [ ] OAuth integration (Google, GitHub) (optional)
- [ ] End-to-end tests

**Deliverable**: Polished UX, feature-complete authentication system.

---

## 8. Testing Strategy

### 8.1 Unit Tests

#### Backend Tests (pytest)
```python
# tests/auth/test_password_hashing.py
def test_password_hash():
    password = "SecurePass123!"
    hashed = hash_password(password)
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

# tests/auth/test_jwt.py
def test_jwt_token_generation():
    user = {"id": 42, "username": "test_user", "role": "user"}
    token = generate_jwt_token(user)
    payload = verify_token(token)
    assert payload["sub"] == 42
    assert payload["username"] == "test_user"

# tests/auth/test_account_lockout.py
async def test_account_lockout():
    # Simulate 5 failed logins
    for _ in range(5):
        await handle_failed_login("test_user", db)

    user = await db.get_user_by_username("test_user")
    assert user.is_locked is True
```

#### Frontend Tests (Vitest + React Testing Library)
```typescript
// tests/components/LoginForm.test.tsx
test('displays error on invalid credentials', async () => {
  render(<LoginForm />);
  fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'invalid' } });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'wrong' } });
  fireEvent.click(screen.getByText('Log In'));

  await waitFor(() => {
    expect(screen.getByText(/invalid username or password/i)).toBeInTheDocument();
  });
});
```

---

### 8.2 Integration Tests

#### API Tests (pytest + httpx)
```python
# tests/integration/test_auth_flow.py
async def test_signup_login_flow():
    # Signup
    response = await client.post("/auth/signup", json={
        "username": "newuser",
        "email": "new@example.com",
        "password": "SecurePass123!",
        "password_confirm": "SecurePass123!",
        "captcha_token": "test_token"
    })
    assert response.status_code == 201
    token = response.json()["token"]

    # Login
    response = await client.post("/auth/login", json={
        "username": "newuser",
        "password": "SecurePass123!"
    })
    assert response.status_code == 200
    assert response.json()["token"] is not None

    # Access protected endpoint
    response = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["user"]["username"] == "newuser"
```

---

### 8.3 Security Tests

#### Penetration Testing Checklist
- [ ] SQL injection attempts (parameterized queries)
- [ ] XSS attempts (input sanitization)
- [ ] CSRF attacks (SameSite cookies, CSRF tokens)
- [ ] Brute force login (rate limiting, account lockout)
- [ ] Token tampering (signature verification)
- [ ] Session hijacking (secure cookies, HTTPS only)
- [ ] Password enumeration (generic error messages)
- [ ] Common password bypass (password policy enforcement)

---

## 9. Configuration

### 9.1 Environment Variables

```bash
# .env (Backend)

# Database
DATABASE_URL=sqlite:///./victor.db
# DATABASE_URL=postgresql://user:pass@localhost/victor

# JWT
JWT_SECRET_KEY=<256-bit-random-key>  # openssl rand -hex 32
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# CAPTCHA
RECAPTCHA_SITE_KEY=<your-site-key>
RECAPTCHA_SECRET_KEY=<your-secret-key>

# Password Policy
PASSWORD_MIN_LENGTH=8
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_DIGIT=true
PASSWORD_REQUIRE_SPECIAL=true
PASSWORD_HISTORY_COUNT=3

# Account Lockout
MAX_FAILED_LOGIN_ATTEMPTS=5
ACCOUNT_LOCKOUT_DURATION_MINUTES=30

# Server
ENVIRONMENT=development  # development, staging, production
HTTPS_ONLY=false         # Set to true in production
```

```bash
# .env (Frontend)

VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_RECAPTCHA_SITE_KEY=<your-site-key>
```

---

## 10. Deployment Checklist

### Pre-Deployment
- [ ] Set strong JWT_SECRET_KEY (never use default)
- [ ] Enable HTTPS_ONLY in production
- [ ] Configure proper CORS origins (no wildcards)
- [ ] Set up database backups
- [ ] Review and adjust password policy
- [ ] Configure CAPTCHA keys
- [ ] Force admin password change on first login
- [ ] Enable audit logging
- [ ] Set up monitoring and alerting

### Post-Deployment
- [ ] Test signup flow end-to-end
- [ ] Test login flow end-to-end
- [ ] Test password change
- [ ] Test account lockout
- [ ] Test CAPTCHA integration
- [ ] Verify HTTPS enforcement
- [ ] Check audit logs are being written
- [ ] Performance test (1000+ concurrent users)
- [ ] Security scan (OWASP ZAP, Burp Suite)

---

## 11. Security Considerations

### Threat Model

| Threat | Mitigation |
|--------|------------|
| **Brute force attacks** | Account lockout (5 attempts), CAPTCHA, rate limiting |
| **Credential stuffing** | CAPTCHA on signup/login, password policy |
| **SQL injection** | Parameterized queries, ORM |
| **XSS** | Input sanitization, CSP headers, React escaping |
| **CSRF** | SameSite cookies, CSRF tokens |
| **Session hijacking** | Secure cookies, HTTPS-only, JWT expiration |
| **Man-in-the-middle** | HTTPS/TLS 1.2+, HSTS headers |
| **Rainbow table attacks** | bcrypt with salt (automatic) |
| **Password reuse** | Password history (last 3) |
| **Weak passwords** | Password strength policy, common password check |

---

## 12. Success Criteria

### Functional
- [ ] Users can signup, login, logout
- [ ] Default admin account exists (admin/admin)
- [ ] Users can change passwords
- [ ] Sessions isolated per user
- [ ] CAPTCHA prevents bot signups
- [ ] Account lockout after 5 failures

### Security
- [ ] Passwords hashed with bcrypt (cost 12)
- [ ] JWT tokens properly signed and validated
- [ ] HTTPS enforced in production
- [ ] No SQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] Audit logging complete

### Performance
- [ ] Login < 500ms
- [ ] Signup < 1s
- [ ] Supports 1000+ concurrent users

### Usability
- [ ] Clear error messages
- [ ] Password strength indicator
- [ ] Responsive design (mobile-friendly)

---

## End of Specification

**Status**: 📋 **Design Complete** - Ready for Implementation

**Next Steps**:
1. Review this specification
2. Get stakeholder approval
3. Begin Phase 1 implementation
4. Iterate based on feedback

**Estimated Total Effort**: 3-4 weeks for complete implementation (Phases 1-3)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-26
**Author**: Architecture Team
**Reviewed By**: [Pending]
