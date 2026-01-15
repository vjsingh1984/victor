# Security Policy

## Security Measures in Victor VS Code Extension

The Victor VS Code extension implements multiple layers of security to protect users from malicious commands and code execution vulnerabilities.

## Input Validation

### Command Validation

All terminal commands are validated before execution using an **allowlist-based approach**:

- **Allowlist Commands**: Only pre-approved commands can be executed (victor, git, npm, python, docker, etc.)
- **Pattern Detection**: Dangerous patterns are blocked (shell metacharacters, command injection, path traversal)
- **Flag Validation**: Command flags are checked against an allowlist of safe options

### Protected Against

The validator prevents:

- **Command Injection**: `;`, `|`, `` ` ``, `$()`, `${}`
- **Path Traversal**: `..` in file paths
- **Hex/Unicode Escapes**: `\x2f`, `\u002f`, etc.
- **Newline Injection**: `\n`, `\r` characters
- **Eval/Exec**: Direct execution of arbitrary code
- **Output Redirection**: `>` operators
- **Background Execution**: `&` operator
- **Pipe to Shell**: `curl ... | bash`

### Validation Examples

```typescript
// ✅ Allowed
TerminalCommandValidator.validateCommand('victor chat --provider anthropic')
TerminalCommandValidator.validateCommand('git status')
TerminalCommandValidator.validateCommand('npm install --save-dev typescript')

// ❌ Blocked
TerminalCommandValidator.validateCommand('victor chat; rm -rf /')
TerminalCommandValidator.validateCommand('victor $(whoami)')
TerminalCommandValidator.validateCommand('eval malicious_code')
```

## Path Sanitization

All file paths are sanitized before use:

- **Shell Metacharacter Removal**: `;`, `|`, `` ` ``, `$`, `()`
- **Command Substitution Removal**: `$(...)`, `${...}`, `` `...` ``
- **Path Traversal Prevention**: Blocks `..` sequences
- **Path Validation**: Checks that paths exist and are directories

### Sanitization Examples

```typescript
// Input: /path/to/file;echo malicious
// Output: /path/to/fileecho malicious

// Input: /path/$(whoami)/file
// Output: /path/file
```

## Workspace Path Validation

Before executing commands in a workspace directory:

1. **Sanitize Path**: Remove dangerous characters
2. **Path Traversal Check**: Block `..` sequences
3. **Existence Check**: Verify path exists on filesystem
4. **Type Check**: Ensure path is a directory, not a file

## Command Approval

For additional safety, the extension supports:

- **Dangerous Command Detection**: Regex patterns for high-risk operations
- **User Approval**: Modal dialogs for command confirmation
- **Approval History**: Track which commands were approved
- **Auto-Approval Rules**: Configure rules for trusted commands

### Configuration

Users can configure approval behavior:

```json
{
  "victor.autonomy.requireApprovalForTerminal": true,
  "victor.autonomy.dangerousCommandPatterns": [
    "rm -rf",
    "sudo",
    "mkfs",
    "dd if=",
    "> /dev/"
  ]
}
```

## Child Process Security

When executing commands with output capture:

- **Validation**: Commands validated before spawning process
- **Timeout**: Processes auto-terminate after timeout (default: 60s)
- **Working Directory**: Explicitly set to validated workspace path
- **Shell Mode**: Uses `shell: true` for command compatibility (with validation)

## Security Best Practices

### For Users

1. **Review Commands**: Always review commands before approving execution
2. **Use Workspaces**: Work in dedicated workspace directories
3. **Check Permissions**: Be cautious with commands requiring `sudo`
4. **Update Regularly**: Keep the extension updated for security patches
5. **Report Issues**: Report security vulnerabilities privately

### For Developers

1. **Validate Input**: Always use `TerminalCommandValidator` before command execution
2. **Sanitize Paths**: Use `sanitizePath()` for user-provided file paths
3. **Use Allowlists**: Never use blacklists for security-critical validation
4. **Log Security Events**: Use output channel to log validation failures
5. **Test Security**: Write tests for injection attempts and edge cases

## Extending the Allowlist

To add custom commands to the allowlist:

```typescript
// Extend command allowlist
TerminalCommandValidator.extendAllowlist(['customcmd', 'mytool']);

// Extend flag allowlist
TerminalCommandValidator.extendFlagAllowlist(['--custom-flag']);
```

## Current Allowlists

### Commands (140+)

Development tools: `victor`, `vic`, `python`, `npm`, `yarn`, `cargo`, `go`, `git`, `make`, etc.

Package managers: `pip`, `npm`, `yarn`, `pnpm`, `cargo`, `gem`, `composer`, etc.

Build tools: `make`, `cmake`, `mvn`, `gradle`, `cargo`, `go build`, etc.

Test tools: `pytest`, `jest`, `mocha`, `unittest`, etc.

Linters: `eslint`, `ruff`, `black`, `mypy`, `flake8`, etc.

Containers: `docker`, `docker-compose`, `kubectl`, `helm`, etc.

### Flags (80+)

Common flags: `--help`, `--version`, `--verbose`, `--quiet`, `--output`, etc.

Victor flags: `--provider`, `--model`, `--mode`, `--profile`, `--no-tui`, etc.

Python flags: `--install`, `--upgrade`, `--editable`, `--requirement`, etc.

Git flags: `--all`, `--branches`, `--oneline`, `--stat`, `--patch`, etc.

## Security Auditing

### Recent Security Improvements

- **2026-01-14**: Added comprehensive input validation (Phase 5.1)
  - Created `TerminalCommandValidator` with allowlist-based validation
  - Added 100+ command injection prevention tests
  - Implemented path sanitization and workspace validation
  - Updated `terminalProvider.ts` to use validation before all command execution

### Security Scan Results

- **Bandit**: No critical issues in Python backend
- **Semgrep**: VS Code extension command injection issue RESOLVED
- **Safety**: All dependency vulnerabilities addressed

## Reporting Vulnerabilities

To report a security vulnerability:

1. **Do NOT** create public issues
2. Email: security@victor-ai.com
3. Include: Description, steps to reproduce, impact assessment
4. We will respond within 48 hours
5. Coordinated disclosure timeline will be agreed upon

## Security Contact

- **Security Email**: security@victor-ai.com
- **PGP Key**: Available on keyserver (request via email)
- **Bug Bounty**: See https://victor-ai.com/security

## Additional Resources

- [OWASP Command Injection](https://owasp.org/www-community/attacks/Command_Injection)
- [Node.js Security Best Practices](https://nodejs.org/en/docs/guides/security-practices/)
- [VS Code Extension Guidelines](https://code.visualstudio.com/api/references/extension-guides/security)

## License

Copyright © 2026 Victor AI. All rights reserved.

Security measures are licensed under the same Apache 2.0 license as the Victor project.
