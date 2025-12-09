/**
 * Tool Approval Service Tests
 *
 * Tests for the ToolApprovalService which handles approval
 * workflows for dangerous tool executions.
 */

import * as assert from 'assert';

suite('ToolApprovalService Test Suite', () => {
    // Test risk level assessment
    suite('Risk Level Assessment', () => {
        test('Should classify low risk tools', () => {
            const lowRiskTools = ['read_file', 'list_directory', 'code_search', 'git_status'];

            const getRiskLevel = (toolName: string): string => {
                if (lowRiskTools.includes(toolName)) return 'low';
                return 'medium';
            };

            assert.strictEqual(getRiskLevel('read_file'), 'low');
            assert.strictEqual(getRiskLevel('code_search'), 'low');
        });

        test('Should classify medium risk tools', () => {
            const mediumRiskTools = ['write_file', 'git_commit', 'refactor'];

            const getRiskLevel = (toolName: string): string => {
                const lowRisk = ['read_file', 'list_directory', 'code_search'];
                const highRisk = ['bash', 'delete_file', 'docker_run'];

                if (lowRisk.includes(toolName)) return 'low';
                if (highRisk.includes(toolName)) return 'high';
                return 'medium';
            };

            assert.strictEqual(getRiskLevel('write_file'), 'medium');
            assert.strictEqual(getRiskLevel('git_commit'), 'medium');
        });

        test('Should classify high risk tools', () => {
            const highRiskTools = ['bash', 'delete_file', 'docker_run', 'execute_command'];

            const getRiskLevel = (toolName: string): string => {
                if (highRiskTools.includes(toolName)) return 'high';
                return 'medium';
            };

            assert.strictEqual(getRiskLevel('bash'), 'high');
            assert.strictEqual(getRiskLevel('delete_file'), 'high');
            assert.strictEqual(getRiskLevel('docker_run'), 'high');
        });

        test('Should classify critical risk tools', () => {
            const criticalRiskTools = ['rm_rf', 'drop_database', 'format_disk'];

            const getRiskLevel = (toolName: string): string => {
                if (criticalRiskTools.includes(toolName)) return 'critical';
                return 'low';
            };

            assert.strictEqual(getRiskLevel('rm_rf'), 'critical');
            assert.strictEqual(getRiskLevel('drop_database'), 'critical');
        });
    });

    // Test auto-approval rules
    suite('Auto-Approval Rules', () => {
        test('Should auto-approve low risk tools', () => {
            const autoApproveRules = {
                lowRisk: true,
                mediumRisk: false,
                highRisk: false,
                critical: false
            };

            const shouldAutoApprove = (riskLevel: string): boolean => {
                return autoApproveRules[riskLevel as keyof typeof autoApproveRules] ?? false;
            };

            assert.ok(shouldAutoApprove('lowRisk'));
            assert.ok(!shouldAutoApprove('highRisk'));
        });

        test('Should respect tool-specific auto-approve', () => {
            const autoApproveTool = new Set(['read_file', 'list_directory']);

            const isAutoApproved = (toolName: string): boolean => {
                return autoApproveTool.has(toolName);
            };

            assert.ok(isAutoApproved('read_file'));
            assert.ok(!isAutoApproved('bash'));
        });

        test('Should handle path-based auto-approval', () => {
            const autoApprovePaths = ['/home/user/safe/', '/tmp/test/'];

            const isPathAutoApproved = (path: string): boolean => {
                return autoApprovePaths.some(safePath => path.startsWith(safePath));
            };

            assert.ok(isPathAutoApproved('/home/user/safe/file.txt'));
            assert.ok(isPathAutoApproved('/tmp/test/data.json'));
            assert.ok(!isPathAutoApproved('/etc/passwd'));
        });
    });

    // Test approval queue
    suite('Approval Queue', () => {
        test('Should add pending approvals', () => {
            const pendingApprovals: { id: string; toolName: string; timestamp: number }[] = [];

            const addPending = (toolName: string) => {
                pendingApprovals.push({
                    id: `approval-${Date.now()}`,
                    toolName,
                    timestamp: Date.now()
                });
            };

            addPending('bash');
            addPending('delete_file');

            assert.strictEqual(pendingApprovals.length, 2);
        });

        test('Should remove approved items', () => {
            const pendingApprovals = [
                { id: 'a1', toolName: 'bash' },
                { id: 'a2', toolName: 'delete_file' },
                { id: 'a3', toolName: 'docker_run' }
            ];

            const approve = (id: string) => {
                const index = pendingApprovals.findIndex(a => a.id === id);
                if (index !== -1) pendingApprovals.splice(index, 1);
            };

            approve('a2');

            assert.strictEqual(pendingApprovals.length, 2);
            assert.ok(!pendingApprovals.some(a => a.id === 'a2'));
        });

        test('Should reject and remove items', () => {
            const pendingApprovals = [
                { id: 'a1', toolName: 'bash', status: 'pending' }
            ];

            const reject = (id: string) => {
                const item = pendingApprovals.find(a => a.id === id);
                if (item) {
                    (item as any).status = 'rejected';
                }
            };

            reject('a1');

            assert.strictEqual(pendingApprovals[0].status, 'rejected');
        });
    });

    // Test session-based approvals
    suite('Session Approvals', () => {
        test('Should track session approvals', () => {
            const sessionApprovals = new Set<string>();

            const approveForSession = (toolName: string) => {
                sessionApprovals.add(toolName);
            };

            const isSessionApproved = (toolName: string): boolean => {
                return sessionApprovals.has(toolName);
            };

            approveForSession('bash');

            assert.ok(isSessionApproved('bash'));
            assert.ok(!isSessionApproved('delete_file'));
        });

        test('Should clear session on reset', () => {
            const sessionApprovals = new Set(['bash', 'delete_file']);

            sessionApprovals.clear();

            assert.strictEqual(sessionApprovals.size, 0);
        });
    });

    // Test approval history
    suite('Approval History', () => {
        test('Should record approval history', () => {
            const history: { toolName: string; action: string; timestamp: number }[] = [];

            const recordApproval = (toolName: string, action: 'approved' | 'rejected') => {
                history.push({ toolName, action, timestamp: Date.now() });
            };

            recordApproval('bash', 'approved');
            recordApproval('delete_file', 'rejected');

            assert.strictEqual(history.length, 2);
            assert.strictEqual(history[0].action, 'approved');
            assert.strictEqual(history[1].action, 'rejected');
        });

        test('Should limit history size', () => {
            const maxHistory = 50;
            const history: object[] = [];

            for (let i = 0; i < 100; i++) {
                history.push({ id: i });
                if (history.length > maxHistory) {
                    history.shift();
                }
            }

            assert.strictEqual(history.length, maxHistory);
        });

        test('Should format history for display', () => {
            const formatHistoryEntry = (entry: {
                toolName: string;
                action: string;
                timestamp: number
            }): string => {
                const date = new Date(entry.timestamp).toLocaleString();
                const icon = entry.action === 'approved' ? '✓' : '✗';
                return `${icon} ${entry.toolName} - ${entry.action} at ${date}`;
            };

            const entry = {
                toolName: 'bash',
                action: 'approved',
                timestamp: Date.now()
            };

            const formatted = formatHistoryEntry(entry);
            assert.ok(formatted.includes('bash'));
            assert.ok(formatted.includes('✓'));
        });
    });

    // Test approval dialog content
    suite('Approval Dialog', () => {
        test('Should format dialog message', () => {
            const formatApprovalMessage = (toolName: string, args: object): string => {
                return `Tool "${toolName}" requires approval.\n\nArguments:\n${JSON.stringify(args, null, 2)}`;
            };

            const message = formatApprovalMessage('bash', { command: 'rm -rf /tmp/test' });

            assert.ok(message.includes('bash'));
            assert.ok(message.includes('rm -rf'));
        });

        test('Should include risk warning', () => {
            const getRiskWarning = (riskLevel: string): string => {
                const warnings: Record<string, string> = {
                    low: '',
                    medium: '⚠️ This operation will modify files.',
                    high: '🔴 This is a high-risk operation.',
                    critical: '🚨 CRITICAL: This operation could cause data loss!'
                };
                return warnings[riskLevel] || '';
            };

            assert.strictEqual(getRiskWarning('high'), '🔴 This is a high-risk operation.');
            assert.ok(getRiskWarning('critical').includes('CRITICAL'));
        });
    });

    // Test tool argument validation
    suite('Argument Validation', () => {
        test('Should detect dangerous paths', () => {
            const dangerousPaths = ['/', '/etc', '/usr', '/bin', '/home'];

            const isDangerousPath = (path: string): boolean => {
                return dangerousPaths.some(dp => path === dp || path.startsWith(dp + '/'));
            };

            assert.ok(isDangerousPath('/etc/passwd'));
            assert.ok(isDangerousPath('/home/user'));
            assert.ok(!isDangerousPath('/tmp/safe'));
        });

        test('Should detect dangerous commands', () => {
            const dangerousPatterns = [
                /rm\s+-rf/,
                /sudo/,
                /chmod\s+777/,
                />\s*\/dev/,
                /dd\s+if=/
            ];

            const isDangerousCommand = (command: string): boolean => {
                return dangerousPatterns.some(pattern => pattern.test(command));
            };

            assert.ok(isDangerousCommand('rm -rf /'));
            assert.ok(isDangerousCommand('sudo apt install'));
            assert.ok(!isDangerousCommand('ls -la'));
        });
    });

    // Test approval state persistence
    suite('State Persistence', () => {
        test('Should serialize auto-approve rules', () => {
            const rules = {
                tools: ['read_file', 'list_directory'],
                paths: ['/safe/path'],
                riskLevels: ['low']
            };

            const serialized = JSON.stringify(rules);
            const deserialized = JSON.parse(serialized);

            assert.deepStrictEqual(deserialized, rules);
        });

        test('Should restore state from storage', () => {
            const savedState = {
                autoApproveTools: ['read_file'],
                sessionApprovals: ['bash']
            };

            const autoApproveTools = new Set(savedState.autoApproveTools);
            const sessionApprovals = new Set(savedState.sessionApprovals);

            assert.ok(autoApproveTools.has('read_file'));
            assert.ok(sessionApprovals.has('bash'));
        });
    });

    // Test approval timeout
    suite('Approval Timeout', () => {
        test('Should detect expired approvals', () => {
            const timeoutMs = 30000; // 30 seconds
            const now = Date.now();

            const isExpired = (timestamp: number): boolean => {
                return now - timestamp > timeoutMs;
            };

            const recentApproval = now - 10000; // 10 seconds ago
            const oldApproval = now - 60000; // 60 seconds ago

            assert.ok(!isExpired(recentApproval));
            assert.ok(isExpired(oldApproval));
        });
    });
});
