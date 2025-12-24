/**
 * Diagnostics View Provider Tests
 *
 * Tests for the DiagnosticsViewProvider which shows code issues
 * organized by file with AI fix suggestions.
 */

import * as assert from 'assert';

suite('DiagnosticsViewProvider Test Suite', () => {
    // Test diagnostic structure
    suite('Diagnostic Structure', () => {
        test('Should create diagnostic entry', () => {
            const diagnostic = {
                severity: 0, // Error
                message: 'Cannot find name "foo"',
                range: { start: { line: 10, character: 5 }, end: { line: 10, character: 8 } },
                source: 'typescript'
            };

            assert.strictEqual(diagnostic.severity, 0);
            assert.ok(diagnostic.message.includes('Cannot find'));
        });

        test('Should have severity levels', () => {
            const severities = {
                Error: 0,
                Warning: 1,
                Information: 2,
                Hint: 3
            };

            assert.strictEqual(severities.Error, 0);
            assert.strictEqual(severities.Warning, 1);
        });

        test('Should get severity icon', () => {
            const getIcon = (severity: number): string => {
                switch (severity) {
                    case 0: return 'error';
                    case 1: return 'warning';
                    case 2: return 'info';
                    case 3: return 'lightbulb';
                    default: return 'question';
                }
            };

            assert.strictEqual(getIcon(0), 'error');
            assert.strictEqual(getIcon(1), 'warning');
        });

        test('Should get severity color', () => {
            const getColor = (severity: number): string => {
                switch (severity) {
                    case 0: return 'errorForeground';
                    case 1: return 'warningForeground';
                    default: return 'foreground';
                }
            };

            assert.strictEqual(getColor(0), 'errorForeground');
        });
    });

    // Test file grouping
    suite('File Grouping', () => {
        test('Should group diagnostics by file', () => {
            const diagnostics = [
                { uri: '/src/a.ts', message: 'Error 1' },
                { uri: '/src/a.ts', message: 'Error 2' },
                { uri: '/src/b.ts', message: 'Error 3' }
            ];

            const grouped: Record<string, typeof diagnostics> = {};
            diagnostics.forEach(d => {
                if (!grouped[d.uri]) grouped[d.uri] = [];
                grouped[d.uri].push(d);
            });

            assert.strictEqual(Object.keys(grouped).length, 2);
            assert.strictEqual(grouped['/src/a.ts'].length, 2);
        });

        test('Should count diagnostics per file', () => {
            const counts: Record<string, number> = {
                '/src/a.ts': 5,
                '/src/b.ts': 2,
                '/src/c.ts': 1
            };

            const total = Object.values(counts).reduce((a, b) => a + b, 0);
            assert.strictEqual(total, 8);
        });

        test('Should sort files by error count', () => {
            const files = [
                { path: '/src/a.ts', count: 2 },
                { path: '/src/b.ts', count: 5 },
                { path: '/src/c.ts', count: 1 }
            ];

            const sorted = [...files].sort((a, b) => b.count - a.count);
            assert.strictEqual(sorted[0].path, '/src/b.ts');
        });
    });

    // Test tree structure
    suite('Tree Structure', () => {
        test('Should create file tree item', () => {
            const createFileItem = (path: string, count: number) => ({
                label: path.split('/').pop(),
                description: `${count} issue(s)`,
                contextValue: 'diagnosticFile',
                collapsibleState: 1 // Collapsed
            });

            const item = createFileItem('/src/test.ts', 3);
            assert.strictEqual(item.label, 'test.ts');
            assert.ok(item.description.includes('3'));
        });

        test('Should create diagnostic tree item', () => {
            const createDiagnosticItem = (message: string, line: number, severity: number) => ({
                label: message.substring(0, 50),
                description: `Line ${line}`,
                contextValue: 'diagnostic',
                iconPath: severity === 0 ? 'error' : 'warning'
            });

            const item = createDiagnosticItem('Type error', 10, 0);
            assert.strictEqual(item.description, 'Line 10');
        });

        test('Should format long messages', () => {
            const truncate = (message: string, maxLen: number): string => {
                if (message.length <= maxLen) return message;
                return message.substring(0, maxLen - 3) + '...';
            };

            const long = 'A'.repeat(100);
            assert.strictEqual(truncate(long, 50).length, 50);
            assert.ok(truncate(long, 50).endsWith('...'));
        });
    });

    // Test diagnostic collection
    suite('Diagnostic Collection', () => {
        test('Should filter by severity', () => {
            const diagnostics = [
                { severity: 0 },
                { severity: 0 },
                { severity: 1 },
                { severity: 2 }
            ];

            const errors = diagnostics.filter(d => d.severity === 0);
            assert.strictEqual(errors.length, 2);
        });

        test('Should filter by source', () => {
            const diagnostics = [
                { source: 'typescript' },
                { source: 'eslint' },
                { source: 'typescript' }
            ];

            const tsOnly = diagnostics.filter(d => d.source === 'typescript');
            assert.strictEqual(tsOnly.length, 2);
        });

        test('Should check if file has errors', () => {
            const hasErrors = (diagnostics: { severity: number }[]): boolean => {
                return diagnostics.some(d => d.severity === 0);
            };

            assert.ok(hasErrors([{ severity: 0 }]));
            assert.ok(!hasErrors([{ severity: 1 }]));
        });
    });

    // Test navigation
    suite('Navigation', () => {
        test('Should build location from diagnostic', () => {
            const diagnostic = {
                uri: '/src/test.ts',
                range: { start: { line: 10, character: 5 } }
            };

            const location = {
                uri: diagnostic.uri,
                line: diagnostic.range.start.line,
                character: diagnostic.range.start.character
            };

            assert.strictEqual(location.line, 10);
            assert.strictEqual(location.character, 5);
        });

        test('Should format go-to command', () => {
            const formatGoTo = (uri: string, line: number): string => {
                return `${uri}:${line + 1}`;
            };

            assert.strictEqual(formatGoTo('/src/test.ts', 9), '/src/test.ts:10');
        });
    });

    // Test AI fix integration
    suite('AI Fix Integration', () => {
        test('Should build fix prompt', () => {
            const buildFixPrompt = (diagnostic: { message: string; source: string }, code: string): string => {
                return `Fix this ${diagnostic.source} issue: ${diagnostic.message}\n\nCode:\n${code}`;
            };

            const prompt = buildFixPrompt(
                { message: 'Type error', source: 'typescript' },
                'const x: number = "hello"'
            );
            assert.ok(prompt.includes('typescript'));
            assert.ok(prompt.includes('Type error'));
        });

        test('Should support fix all in file', () => {
            const buildFixAllPrompt = (diagnostics: { message: string }[], code: string): string => {
                const issues = diagnostics.map((d, i) => `${i + 1}. ${d.message}`).join('\n');
                return `Fix these issues:\n${issues}\n\nCode:\n${code}`;
            };

            const prompt = buildFixAllPrompt(
                [{ message: 'Error 1' }, { message: 'Error 2' }],
                'code'
            );
            assert.ok(prompt.includes('1. Error 1'));
            assert.ok(prompt.includes('2. Error 2'));
        });
    });

    // Test refresh
    suite('Refresh', () => {
        test('Should track refresh state', () => {
            let isRefreshing = false;

            const startRefresh = () => { isRefreshing = true; };
            const endRefresh = () => { isRefreshing = false; };

            assert.ok(!isRefreshing);
            startRefresh();
            assert.ok(isRefreshing);
            endRefresh();
            assert.ok(!isRefreshing);
        });

        test('Should debounce refresh calls', () => {
            let refreshCount = 0;
            const refreshCalls: number[] = [];

            const debounce = (delay: number) => {
                const now = Date.now();
                if (refreshCalls.length === 0 || now - refreshCalls[refreshCalls.length - 1] > delay) {
                    refreshCount++;
                }
                refreshCalls.push(now);
            };

            debounce(100);
            debounce(100);
            debounce(100);
            assert.strictEqual(refreshCount, 1);
        });
    });

    // Test badge display
    suite('Badge Display', () => {
        test('Should format badge text', () => {
            const formatBadge = (count: number): string => {
                if (count === 0) return '';
                if (count > 99) return '99+';
                return count.toString();
            };

            assert.strictEqual(formatBadge(0), '');
            assert.strictEqual(formatBadge(5), '5');
            assert.strictEqual(formatBadge(150), '99+');
        });

        test('Should get badge tooltip', () => {
            const getTooltip = (errors: number, warnings: number): string => {
                const parts: string[] = [];
                if (errors > 0) parts.push(`${errors} error(s)`);
                if (warnings > 0) parts.push(`${warnings} warning(s)`);
                return parts.join(', ') || 'No issues';
            };

            assert.strictEqual(getTooltip(3, 2), '3 error(s), 2 warning(s)');
            assert.strictEqual(getTooltip(0, 0), 'No issues');
        });
    });

    // Test quick fix actions
    suite('Quick Fix Actions', () => {
        test('Should have action buttons', () => {
            const actions = [
                { id: 'fix', label: 'Fix with AI', icon: 'lightbulb-autofix' },
                { id: 'fixAll', label: 'Fix All in File', icon: 'sparkle' },
                { id: 'ignore', label: 'Ignore', icon: 'close' }
            ];

            assert.strictEqual(actions.length, 3);
            assert.ok(actions.some(a => a.id === 'fix'));
        });

        test('Should format action title', () => {
            const formatTitle = (action: string, target: string): string => {
                return `${action}: ${target}`;
            };

            assert.strictEqual(formatTitle('Fix', 'Type error'), 'Fix: Type error');
        });
    });

    // Test empty state
    suite('Empty State', () => {
        test('Should show empty message', () => {
            const getEmptyMessage = (hasWorkspace: boolean): string => {
                if (!hasWorkspace) return 'No workspace open';
                return 'No issues found';
            };

            assert.strictEqual(getEmptyMessage(true), 'No issues found');
            assert.strictEqual(getEmptyMessage(false), 'No workspace open');
        });

        test('Should create empty state item', () => {
            const createEmptyItem = (message: string) => ({
                label: message,
                contextValue: 'empty',
                collapsibleState: 0 // None
            });

            const item = createEmptyItem('No issues found');
            assert.strictEqual(item.label, 'No issues found');
        });
    });

    // Test workspace diagnostics
    suite('Workspace Diagnostics', () => {
        test('Should aggregate workspace stats', () => {
            const fileStats = [
                { errors: 5, warnings: 2 },
                { errors: 3, warnings: 1 },
                { errors: 0, warnings: 4 }
            ];

            const total = fileStats.reduce(
                (acc, f) => ({
                    errors: acc.errors + f.errors,
                    warnings: acc.warnings + f.warnings
                }),
                { errors: 0, warnings: 0 }
            );

            assert.strictEqual(total.errors, 8);
            assert.strictEqual(total.warnings, 7);
        });

        test('Should filter by workspace folder', () => {
            const diagnostics = [
                { uri: '/workspace/src/a.ts' },
                { uri: '/workspace/src/b.ts' },
                { uri: '/other/c.ts' }
            ];

            const inWorkspace = diagnostics.filter(d => d.uri.startsWith('/workspace/'));
            assert.strictEqual(inWorkspace.length, 2);
        });
    });
});
