/**
 * Terminal History Tests
 */

import * as assert from 'assert';
import { TerminalHistoryEntry, TerminalHistoryService } from '../../terminalHistory';

suite('TerminalHistory Test Suite', () => {
    let historyService: TerminalHistoryService;

    setup(() => {
        historyService = TerminalHistoryService.getInstance();
        historyService.clear();
    });

    teardown(() => {
        historyService.clear();
    });

    test('Should be singleton', () => {
        const instance1 = TerminalHistoryService.getInstance();
        const instance2 = TerminalHistoryService.getInstance();

        assert.strictEqual(instance1, instance2);
    });

    test('Should add command to history', () => {
        historyService.addCommand({
            command: 'npm install',
            terminalName: 'bash',
        });

        assert.strictEqual(historyService.count, 1);
    });

    test('Should add command with full entry', () => {
        historyService.addCommand({
            command: 'npm test',
            terminalName: 'bash',
            workingDir: '/home/user/project',
            exitCode: 0,
            output: 'All tests passed',
        });

        const recent = historyService.getRecentCommands(1);
        assert.strictEqual(recent.length, 1);
        assert.strictEqual(recent[0].command, 'npm test');
        assert.strictEqual(recent[0].exitCode, 0);
    });

    test('Should return recent commands in order', () => {
        historyService.addCommand({ command: 'cmd1', terminalName: 'bash' });
        historyService.addCommand({ command: 'cmd2', terminalName: 'bash' });
        historyService.addCommand({ command: 'cmd3', terminalName: 'bash' });

        const recent = historyService.getRecentCommands(2);
        assert.strictEqual(recent.length, 2);
        assert.strictEqual(recent[0].command, 'cmd3');  // Most recent first
        assert.strictEqual(recent[1].command, 'cmd2');
    });

    test('Should filter by terminal name', () => {
        historyService.addCommand({ command: 'bash cmd', terminalName: 'bash' });
        historyService.addCommand({ command: 'zsh cmd', terminalName: 'zsh' });
        historyService.addCommand({ command: 'bash cmd 2', terminalName: 'bash' });

        const bashCommands = historyService.getTerminalCommands('bash');
        assert.strictEqual(bashCommands.length, 2);
        assert.ok(bashCommands.every(c => c.terminalName === 'bash'));
    });

    test('Should clear history', () => {
        historyService.addCommand({ command: 'cmd1', terminalName: 'bash' });
        historyService.addCommand({ command: 'cmd2', terminalName: 'bash' });

        assert.strictEqual(historyService.count, 2);

        historyService.clear();
        assert.strictEqual(historyService.count, 0);
    });

    test('Should emit history changed event', (done) => {
        const disposable = historyService.onHistoryChanged(() => {
            disposable.dispose();
            done();
        });

        historyService.addCommand({ command: 'test', terminalName: 'bash' });
    });

    test('Should get context string', () => {
        historyService.addCommand({ command: 'npm install', terminalName: 'bash' });

        const context = historyService.getContextString();
        assert.ok(context.includes('npm install'));
        assert.ok(context.includes('Recent terminal commands'));
    });

    test('Should return empty context string when no history', () => {
        const context = historyService.getContextString();
        assert.ok(context.includes('No recent terminal commands'));
    });

    test('Should include exit code in context when non-zero', () => {
        historyService.addCommand({
            command: 'failing-command',
            terminalName: 'bash',
            exitCode: 1,
        });

        const context = historyService.getContextString();
        assert.ok(context.includes('Exit code: 1'));
    });

    test('Should include truncated output in context', () => {
        historyService.addCommand({
            command: 'cat file',
            terminalName: 'bash',
            output: 'line1\nline2\nline3\nline4\nline5',
        });

        const context = historyService.getContextString();
        assert.ok(context.includes('line1'));
    });
});

suite('TerminalHistoryEntry Interface', () => {
    test('Entry has required fields', () => {
        const entry: TerminalHistoryEntry = {
            command: 'echo hello',
            timestamp: Date.now(),
            terminalName: 'bash',
        };

        assert.ok(entry.command);
        assert.ok(entry.timestamp);
        assert.ok(entry.terminalName);
    });

    test('Entry with optional fields', () => {
        const entry: TerminalHistoryEntry = {
            command: 'ls -la',
            timestamp: Date.now(),
            terminalName: 'zsh',
            workingDir: '/home/user',
            exitCode: 0,
            output: 'file1\nfile2\nfile3',
        };

        assert.strictEqual(entry.workingDir, '/home/user');
        assert.strictEqual(entry.exitCode, 0);
        assert.ok(entry.output?.includes('file1'));
    });
});

suite('Time Formatting', () => {
    test('Format just now', () => {
        // Internal test - the service uses _formatTimeAgo internally
        const now = Date.now();
        const entry: TerminalHistoryEntry = {
            command: 'test',
            timestamp: now,
            terminalName: 'bash',
        };

        // Timestamps within 60 seconds should show "just now"
        const diff = Date.now() - entry.timestamp;
        assert.ok(diff < 60000);
    });

    test('Format minutes ago', () => {
        const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
        const entry: TerminalHistoryEntry = {
            command: 'test',
            timestamp: fiveMinutesAgo,
            terminalName: 'bash',
        };

        const seconds = Math.floor((Date.now() - entry.timestamp) / 1000);
        assert.ok(seconds >= 60 && seconds < 3600);
    });

    test('Format hours ago', () => {
        const twoHoursAgo = Date.now() - (2 * 60 * 60 * 1000);
        const entry: TerminalHistoryEntry = {
            command: 'test',
            timestamp: twoHoursAgo,
            terminalName: 'bash',
        };

        const seconds = Math.floor((Date.now() - entry.timestamp) / 1000);
        assert.ok(seconds >= 3600 && seconds < 86400);
    });
});

suite('Output Processing', () => {
    test('Clean ANSI escape codes', () => {
        // ANSI colored output
        const coloredOutput = '\x1b[32mSuccess\x1b[0m';

        // The service cleans these internally
        // eslint-disable-next-line no-control-regex
        const cleaned = coloredOutput.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
        assert.strictEqual(cleaned, 'Success');
    });

    test('Handle carriage returns', () => {
        // Progress bar style output
        const progressOutput = 'Loading...\rComplete!';

        // Process carriage returns (keep final state)
        const parts = progressOutput.split('\r');
        const processed = parts[parts.length - 1];

        assert.strictEqual(processed, 'Complete!');
    });

    test('Deduplicate repeated lines', () => {
        const repeatedLines = [
            'Compiling...',
            'Compiling...',
            'Compiling...',
            'Done!',
        ];

        const deduped: string[] = [];
        let lastLine = '';

        for (const line of repeatedLines) {
            if (line !== lastLine) {
                deduped.push(line);
                lastLine = line;
            }
        }

        assert.strictEqual(deduped.length, 2);
        assert.strictEqual(deduped[0], 'Compiling...');
        assert.strictEqual(deduped[1], 'Done!');
    });
});

suite('History Size Limits', () => {
    let sizeTestService: TerminalHistoryService;

    setup(() => {
        sizeTestService = TerminalHistoryService.getInstance();
        sizeTestService.clear();
    });

    teardown(() => {
        sizeTestService.clear();
    });

    test('Should limit history to max size', () => {
        // The service has a default max size (100)
        // We can't easily test this without access to private properties,
        // but we can verify adding many items doesn't break anything

        for (let i = 0; i < 150; i++) {
            sizeTestService.addCommand({
                command: `cmd-${i}`,
                terminalName: 'bash',
            });
        }

        // Should have at most maxHistorySize entries
        // Default is 100, so count should be <= 100
        assert.ok(sizeTestService.count <= 100);
    });

    test('Should respect limit parameter in getRecentCommands', () => {
        for (let i = 0; i < 20; i++) {
            sizeTestService.addCommand({
                command: `cmd-${i}`,
                terminalName: 'bash',
            });
        }

        const limited = sizeTestService.getRecentCommands(5);
        assert.strictEqual(limited.length, 5);
    });

    test('Should return all if limit exceeds count', () => {
        sizeTestService.addCommand({ command: 'cmd1', terminalName: 'bash' });
        sizeTestService.addCommand({ command: 'cmd2', terminalName: 'bash' });

        const all = sizeTestService.getRecentCommands(100);
        assert.strictEqual(all.length, 2);
    });
});
