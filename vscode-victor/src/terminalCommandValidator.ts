/**
 * Terminal Command Validator
 *
 * Provides input validation and sanitization for terminal commands
 * to prevent command injection attacks.
 *
 * Security Features:
 * - Allowlist-based command validation
 * - Shell metacharacter detection
 * - Path traversal prevention
 * - Workspace path validation
 */

import * as fs from 'fs';
import * as path from 'path';

/**
 * Validation error with details
 */
export class ValidationError extends Error {
    constructor(
        message: string,
        public readonly category: 'command' | 'path' | 'pattern' | 'workspace',
        public readonly details?: string
    ) {
        super(message);
        this.name = 'ValidationError';
    }
}

/**
 * Validation result
 */
export interface ValidationResult {
    valid: boolean;
    error?: ValidationError;
    sanitized?: string;
}

/**
 * Terminal command validator with allowlist-based security
 */
export class TerminalCommandValidator {
    // Allowlist of safe base commands
    private static readonly ALLOWED_COMMANDS = [
        'victor',
        'vic',
        'python',
        'python3',
        'pip',
        'pip3',
        'node',
        'npm',
        'npx',
        'yarn',
        'pnpm',
        'git',
        'make',
        'cmake',
        'cargo',
        'go',
        'rustc',
        'java',
        'javac',
        'mvn',
        'gradle',
        'dotnet',
        'nuget',
        'ruby',
        'gem',
        'php',
        'composer',
        'swift',
        'brew',
        'apt',
        'apt-get',
        'yum',
        'dnf',
        'pacman',
        'curl',
        'wget',
        'tar',
        'zip',
        'unzip',
        'gzip',
        'gunzip',
        'cat',
        'less',
        'more',
        'head',
        'tail',
        'grep',
        'egrep',
        'fgrep',
        'sed',
        'awk',
        'find',
        'locate',
        'ls',
        'dir',
        'cd',
        'pwd',
        'mkdir',
        'rmdir',
        'cp',
        'mv',
        'rm',
        'touch',
        'chmod',
        'chown',
        'ln',
        'df',
        'du',
        'free',
        'top',
        'htop',
        'ps',
        'kill',
        'pkill',
        'killall',
        'systemctl',
        'service',
        'journalctl',
        'systemd-analyze',
        'docker',
        'docker-compose',
        'kubectl',
        'helm',
        'terraform',
        'ansible',
        'pytest',
        'unittest',
        'jest',
        'mocha',
        'jasmine',
        'karma',
        'eslint',
        'prettier',
        'black',
        'ruff',
        'mypy',
        'pylint',
        'flake8',
        'pycodestyle',
        'pydocstyle',
        'bandit',
        'safety',
        'shellcheck',
        'hadolint',
        'tflint',
        'tslint',
        'golangci-lint',
        'rubocop',
        'phpstan',
        'phpcs',
        'eslint',
        'jshint',
        'jscs',
    ];

    // Allowlist of safe flags (can be extended via configuration)
    private static readonly ALLOWED_FLAGS = [
        // Common help/version flags
        '--help',
        '-h',
        '--version',
        '-v',
        '-V',
        '--verbose',
        '-vv',
        '-vvv',

        // Victor-specific flags
        '--no-tui',
        '--provider',
        '--model',
        '--mode',
        '--profile',
        '--temperature',
        '--max-tokens',
        '--timeout',
        '--debug',
        '--quiet',
        '--output',
        '-o',

        // Python/pip flags
        '--install',
        '--upgrade',
        '--uninstall',
        '--list',
        '--show',
        '--freeze',
        '--check',
        '--editable',
        '-e',
        '--requirement',
        '-r',
        '--no-deps',
        '--user',
        '--break-system-packages',
        '--target',
        '--prefix',
        '--root',
        '--upgrade-strategy',
        '--dry-run',
        '--ignore-installed',

        // Git flags
        '--all',
        '--branches',
        '--tags',
        '--remotes',
        '--graph',
        '--oneline',
        '--decorate',
        '--stat',
        '--patch',
        '--name-only',
        '--name-status',
        '--abbrev-commit',
        '--abbrev',
        '--relative',
        '--parents',
        '--left-right',
        '--show-signature',
        '--patch-with-stat',
        '--pretty',

        // npm/yarn flags
        '--save',
        '--save-dev',
        '--save-optional',
        '--save-exact',
        '--global',
        '-g',
        '--force',
        '--production',
        '--development',
        '--only',
        '--also',

        // Test framework flags
        '--verbose',
        '--quiet',
        '--failfast',
        '--catch',
        '--buffer',
        '--traceback',
        '--durations',
        '--cov',
        '--cover',
        '--watch',
        '--bail',
        '--timeout',
        '--reporter',
        '--reporters',

        // Build tool flags
        '--build',
        '--clean',
        '--config',
        '--debug',
        '--release',
        '--target',
        '--output',
        '--optimize',
        '--parallel',
        '--jobs',
        '-j',

        // Common boolean flags
        '--yes',
        '-y',
        '--no',
        '-n',
        '--confirm',
        '--force',
        '-f',
        '--silent',
        '-s',
        '--interactive',
        '-i',
        '--recursive',
        '-r',
        '--verbose',
        '-v',
        '--quiet',
        '-q',
        '--dry-run',
        '--ignore-case',
        '-i',
        '--invert-match',
        '-v',
        '--line-number',
        '-n',
        '--count',
        '-c',
        '--color',
        '--no-color',
        '--watch',
        '-w',
        '--follow',
        '-f',
    ];

    // Dangerous patterns that are NEVER allowed
    private static readonly DANGEROUS_PATTERNS = [
        // Command injection metacharacters
        /[;&|`$()]/,

        // Command substitution
        /\$\([^)]*\)/,
        /\$\{[^}]*\}/,
        /`[^`]*`/,

        // Path traversal attempts
        /\.\.[\/\\]/,
        /\.\.%2f/i,
        /\.\.%5c/i,

        // Hex escape sequences
        /\\x[0-9a-f]{2}/i,

        // Unicode escape sequences
        /\\u[0-9a-f]{4}/i,
        /\\U[0-9a-f]{8}/i,

        // Newline injection
        /\n/,
        /\r/,

        // Tab injection
        /\t/,

        // Multiple consecutive spaces might indicate obfuscation
        { pattern: / {4,}/, severity: 'warning' },

        // Comment characters in command (potential obfuscation)
        /#.*[;&|`$()]/,

        // Redirect operators (shell features)
        />/,

        // Background execution
        /&$/,

        // Command chaining
        /;\s*\w/,

        // Pipe to shell
        /\|\s*(ba)?sh\b/i,

        // Eval attempts
        /\beval\b/i,

        // Exec attempts
        /\bexec\b/i,
    ];

    /**
     * Validates a command string before execution
     *
     * @param command - Command string to validate
     * @throws ValidationError if command is unsafe
     */
    static validateCommand(command: string): ValidationResult {
        // Trim whitespace
        const trimmed = command.trim();

        // Empty command check
        if (!trimmed) {
            return {
                valid: false,
                error: new ValidationError('Command cannot be empty', 'command')
            };
        }

        // Check for dangerous patterns
        for (const patternCheck of this.DANGEROUS_PATTERNS) {
            const pattern = typeof patternCheck === 'object' && 'pattern' in patternCheck
                ? patternCheck.pattern
                : patternCheck;

            if (pattern.test(trimmed)) {
                return {
                    valid: false,
                    error: new ValidationError(
                        `Dangerous pattern detected in command: ${pattern}`,
                        'pattern',
                        `Pattern matched: ${pattern.toString()}`
                    )
                };
            }
        }

        // Extract base command (first word before space)
        const commandParts = trimmed.split(/\s+/);
        const baseCommand = commandParts[0];

        // Validate command is in allowlist
        if (!this.ALLOWED_COMMANDS.includes(baseCommand)) {
            return {
                valid: false,
                error: new ValidationError(
                    `Command not in allowlist: ${baseCommand}`,
                    'command',
                    `Allowed commands: ${this.ALLOWED_COMMANDS.slice(0, 10).join(', ')}...`
                )
            };
        }

        // Validate flags and arguments
        for (let i = 1; i < commandParts.length; i++) {
            const part = commandParts[i];

            // Check if it's a flag
            if (part.startsWith('--')) {
                const flagName = part.split('=')[0];

                if (!this.ALLOWED_FLAGS.includes(flagName)) {
                    // It might be a flag with a value (e.g., --provider anthropic)
                    // Allow it but log a warning (in production, would use telemetry)
                    // For now, we'll be permissive with unknown flags
                    // but ensure they don't contain dangerous patterns
                    if (this._containsDangerousChars(flagName)) {
                        return {
                            valid: false,
                            error: new ValidationError(
                                `Flag contains dangerous characters: ${flagName}`,
                                'pattern'
                            )
                        };
                    }
                }
            } else if (part.startsWith('-')) {
                // Short flags (e.g., -v, -rf)
                // These are harder to validate, so just check for dangerous chars
                if (this._containsDangerousChars(part)) {
                    return {
                        valid: false,
                        error: new ValidationError(
                            `Flag contains dangerous characters: ${part}`,
                            'pattern'
                        )
                    };
                }
            } else {
                // It's an argument (path, value, etc.)
                // Just check for dangerous characters
                if (this._containsDangerousChars(part)) {
                    return {
                        valid: false,
                        error: new ValidationError(
                            `Argument contains dangerous characters: ${part}`,
                            'pattern'
                        )
                    };
                }
            }
        }

        return { valid: true };
    }

    /**
     * Sanitizes a file path by removing dangerous characters
     *
     * @param input - Path to sanitize
     * @returns Sanitized path
     */
    static sanitizePath(input: string): string {
        // Remove shell metacharacters
        let sanitized = input.replace(/[;&|`$()]/g, '');

        // Remove newlines and tabs
        sanitized = sanitized.replace(/[\n\r\t]/g, '');

        // Remove potential command substitution
        sanitized = sanitized.replace(/\$\([^)]*\)/g, '');
        sanitized = sanitized.replace(/\$\{[^}]*\}/g, '');
        sanitized = sanitized.replace(/`[^`]*`/g, '');

        return sanitized;
    }

    /**
     * Validates a workspace path
     *
     * @param workspacePath - Path to validate
     * @returns True if path is valid
     */
    static validateWorkspacePath(workspacePath: string): ValidationResult {
        try {
            // Sanitize the path first
            const sanitized = this.sanitizePath(workspacePath);

            // Check for path traversal
            if (sanitized.includes('..')) {
                return {
                    valid: false,
                    error: new ValidationError(
                        'Path traversal detected',
                        'path',
                        'Path contains ".." which is not allowed'
                    )
                };
            }

            // Resolve to absolute path
            const resolvedPath = path.resolve(sanitized);

            // Check if path exists
            if (!fs.existsSync(resolvedPath)) {
                return {
                    valid: false,
                    error: new ValidationError(
                        'Path does not exist',
                        'workspace',
                        `Path not found: ${resolvedPath}`
                    )
                };
            }

            // Check if it's a directory
            const stats = fs.statSync(resolvedPath);
            if (!stats.isDirectory()) {
                return {
                    valid: false,
                    error: new ValidationError(
                        'Path is not a directory',
                        'workspace',
                        `Expected directory, got file: ${resolvedPath}`
                    )
                };
            }

            return { valid: true, sanitized: resolvedPath };
        } catch (error) {
            return {
                valid: false,
                error: new ValidationError(
                    `Failed to validate workspace path: ${error instanceof Error ? error.message : String(error)}`,
                    'workspace',
                    workspacePath
                )
            };
        }
    }

    /**
     * Validates command arguments
     *
     * @param args - Arguments to validate
     * @returns Validation result
     */
    static validateArguments(args: string[]): ValidationResult {
        for (const arg of args) {
            if (this._containsDangerousChars(arg)) {
                return {
                    valid: false,
                    error: new ValidationError(
                        `Argument contains dangerous characters: ${arg}`,
                        'pattern'
                    )
                };
            }
        }

        return { valid: true };
    }

    /**
     * Checks if a string contains dangerous characters
     *
     * @param input - String to check
     * @returns True if dangerous characters found
     */
    private static _containsDangerousChars(input: string): boolean {
        // Check for shell metacharacters
        const dangerousChars = /[;&|`$()]/;
        return dangerousChars.test(input);
    }

    /**
     * Extends the allowlist with custom commands
     *
     * @param commands - Commands to add to allowlist
     */
    static extendAllowlist(commands: string[]): void {
        for (const cmd of commands) {
            if (!this.ALLOWED_COMMANDS.includes(cmd)) {
                this.ALLOWED_COMMANDS.push(cmd);
            }
        }
    }

    /**
     * Extends the flag allowlist
     *
     * @param flags - Flags to add to allowlist
     */
    static extendFlagAllowlist(flags: string[]): void {
        for (const flag of flags) {
            if (!this.ALLOWED_FLAGS.includes(flag)) {
                this.ALLOWED_FLAGS.push(flag);
            }
        }
    }

    /**
     * Gets the current allowlist (for debugging/configuration UI)
     *
     * @returns Copy of the commands allowlist
     */
    static getAllowlist(): string[] {
        return [...this.ALLOWED_COMMANDS];
    }

    /**
     * Gets the current flag allowlist
     *
     * @returns Copy of the flags allowlist
     */
    static getFlagAllowlist(): string[] {
        return [...this.ALLOWED_FLAGS];
    }
}
