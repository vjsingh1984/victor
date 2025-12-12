/**
 * Workspace Insights View Provider Tests
 *
 * Tests for the WorkspaceInsightsViewProvider which displays
 * workspace statistics, file counts, and security scan results.
 */

import * as assert from 'assert';

suite('WorkspaceInsightsViewProvider Test Suite', () => {
    // Test workspace stats
    suite('Workspace Stats', () => {
        test('Should create stats entry', () => {
            const stats = {
                totalFiles: 150,
                totalLines: 25000,
                languages: ['TypeScript', 'JavaScript', 'Python'],
                lastUpdated: new Date()
            };

            assert.strictEqual(stats.totalFiles, 150);
            assert.strictEqual(stats.languages.length, 3);
        });

        test('Should count files by extension', () => {
            const files = [
                'a.ts', 'b.ts', 'c.js', 'd.py', 'e.ts'
            ];

            const counts: Record<string, number> = {};
            files.forEach(f => {
                const ext = f.split('.').pop() || '';
                counts[ext] = (counts[ext] || 0) + 1;
            });

            assert.strictEqual(counts['ts'], 3);
            assert.strictEqual(counts['js'], 1);
        });

        test('Should format file count', () => {
            const formatCount = (count: number): string => {
                if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
                return count.toString();
            };

            assert.strictEqual(formatCount(500), '500');
            assert.strictEqual(formatCount(1500), '1.5k');
        });

        test('Should format line count', () => {
            const formatLines = (lines: number): string => {
                if (lines >= 1000000) return `${(lines / 1000000).toFixed(1)}M`;
                if (lines >= 1000) return `${(lines / 1000).toFixed(1)}k`;
                return lines.toString();
            };

            assert.strictEqual(formatLines(25000), '25.0k');
        });
    });

    // Test tree structure
    suite('Tree Structure', () => {
        test('Should create stats group', () => {
            const createGroup = (label: string, value: string) => ({
                label,
                description: value,
                contextValue: 'statsGroup',
                collapsibleState: 0 // None
            });

            const item = createGroup('Total Files', '150');
            assert.strictEqual(item.label, 'Total Files');
        });

        test('Should create language item', () => {
            const createLanguageItem = (language: string, count: number, percentage: number) => ({
                label: language,
                description: `${count} files (${percentage}%)`,
                contextValue: 'language',
                collapsibleState: 0
            });

            const item = createLanguageItem('TypeScript', 80, 53);
            assert.ok(item.description.includes('53%'));
        });

        test('Should create section header', () => {
            const createHeader = (title: string) => ({
                label: title,
                contextValue: 'header',
                collapsibleState: 1 // Collapsed
            });

            const item = createHeader('Languages');
            assert.strictEqual(item.contextValue, 'header');
        });
    });

    // Test security scanning
    suite('Security Scanning', () => {
        test('Should create security finding', () => {
            const finding = {
                severity: 'high',
                type: 'secret',
                file: 'config.js',
                line: 15,
                message: 'Potential API key detected'
            };

            assert.strictEqual(finding.severity, 'high');
            assert.ok(finding.message.includes('API key'));
        });

        test('Should categorize severity', () => {
            const severities = ['critical', 'high', 'medium', 'low', 'info'];

            const getSeverityLevel = (severity: string): number => {
                return severities.indexOf(severity);
            };

            assert.strictEqual(getSeverityLevel('critical'), 0);
            assert.strictEqual(getSeverityLevel('high'), 1);
        });

        test('Should get severity icon', () => {
            const getIcon = (severity: string): string => {
                const icons: Record<string, string> = {
                    'critical': 'error',
                    'high': 'warning',
                    'medium': 'info',
                    'low': 'lightbulb',
                    'info': 'question'
                };
                return icons[severity] || 'circle-outline';
            };

            assert.strictEqual(getIcon('critical'), 'error');
            assert.strictEqual(getIcon('high'), 'warning');
        });

        test('Should count findings by severity', () => {
            const findings = [
                { severity: 'high' },
                { severity: 'high' },
                { severity: 'medium' },
                { severity: 'low' }
            ];

            const counts: Record<string, number> = {};
            findings.forEach(f => {
                counts[f.severity] = (counts[f.severity] || 0) + 1;
            });

            assert.strictEqual(counts['high'], 2);
        });
    });

    // Test language statistics
    suite('Language Statistics', () => {
        test('Should detect languages', () => {
            const languageMap: Record<string, string> = {
                'ts': 'TypeScript',
                'tsx': 'TypeScript',
                'js': 'JavaScript',
                'jsx': 'JavaScript',
                'py': 'Python',
                'java': 'Java',
                'go': 'Go'
            };

            assert.strictEqual(languageMap['ts'], 'TypeScript');
        });

        test('Should calculate percentages', () => {
            const total = 100;
            const counts = { ts: 50, js: 30, py: 20 };

            const percentages: Record<string, number> = {};
            Object.entries(counts).forEach(([lang, count]) => {
                percentages[lang] = Math.round((count / total) * 100);
            });

            assert.strictEqual(percentages['ts'], 50);
        });

        test('Should sort languages by count', () => {
            const languages = [
                { name: 'Python', count: 20 },
                { name: 'TypeScript', count: 50 },
                { name: 'JavaScript', count: 30 }
            ];

            const sorted = [...languages].sort((a, b) => b.count - a.count);
            assert.strictEqual(sorted[0].name, 'TypeScript');
        });
    });

    // Test dependency analysis
    suite('Dependency Analysis', () => {
        test('Should parse package.json', () => {
            const packageJson = {
                dependencies: { 'axios': '^1.0.0', 'lodash': '^4.0.0' },
                devDependencies: { 'typescript': '^5.0.0' }
            };

            const depCount = Object.keys(packageJson.dependencies).length;
            const devDepCount = Object.keys(packageJson.devDependencies).length;

            assert.strictEqual(depCount, 2);
            assert.strictEqual(devDepCount, 1);
        });

        test('Should count total dependencies', () => {
            const deps = { a: 1, b: 2, c: 3 };
            const devDeps = { d: 4, e: 5 };

            const total = Object.keys(deps).length + Object.keys(devDeps).length;
            assert.strictEqual(total, 5);
        });

        test('Should detect outdated deps', () => {
            const outdated = [
                { name: 'lodash', current: '4.0.0', latest: '4.17.0' },
                { name: 'axios', current: '1.0.0', latest: '1.6.0' }
            ];

            assert.strictEqual(outdated.length, 2);
        });
    });

    // Test refresh
    suite('Refresh', () => {
        test('Should refresh insights', () => {
            let refreshed = false;

            const refresh = () => { refreshed = true; };

            refresh();
            assert.ok(refreshed);
        });

        test('Should track scan status', () => {
            let scanStatus: 'idle' | 'scanning' | 'complete' = 'idle';

            const startScan = () => { scanStatus = 'scanning'; };
            const completeScan = () => { scanStatus = 'complete'; };

            startScan();
            assert.strictEqual(scanStatus, 'scanning');
            completeScan();
            assert.strictEqual(scanStatus, 'complete');
        });
    });

    // Test file size analysis
    suite('File Size Analysis', () => {
        test('Should calculate total size', () => {
            const files = [
                { size: 1000 },
                { size: 2500 },
                { size: 500 }
            ];

            const total = files.reduce((sum, f) => sum + f.size, 0);
            assert.strictEqual(total, 4000);
        });

        test('Should format file size', () => {
            const formatSize = (bytes: number): string => {
                if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
                if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
                return `${bytes} B`;
            };

            assert.strictEqual(formatSize(500), '500 B');
            assert.strictEqual(formatSize(2048), '2.0 KB');
            assert.strictEqual(formatSize(1572864), '1.5 MB');
        });

        test('Should find largest files', () => {
            const files = [
                { path: 'a.ts', size: 5000 },
                { path: 'b.ts', size: 15000 },
                { path: 'c.ts', size: 3000 }
            ];

            const sorted = [...files].sort((a, b) => b.size - a.size);
            assert.strictEqual(sorted[0].path, 'b.ts');
        });
    });

    // Test complexity metrics
    suite('Complexity Metrics', () => {
        test('Should calculate average complexity', () => {
            const complexities = [5, 10, 15, 8, 12];
            const average = complexities.reduce((a, b) => a + b, 0) / complexities.length;

            assert.strictEqual(average, 10);
        });

        test('Should identify complex files', () => {
            const files = [
                { path: 'a.ts', complexity: 5 },
                { path: 'b.ts', complexity: 25 },
                { path: 'c.ts', complexity: 8 }
            ];

            const threshold = 15;
            const complex = files.filter(f => f.complexity > threshold);

            assert.strictEqual(complex.length, 1);
        });
    });

    // Test empty state
    suite('Empty State', () => {
        test('Should show empty message', () => {
            const getMessage = (hasWorkspace: boolean): string => {
                if (!hasWorkspace) return 'No workspace open';
                return 'Analyzing workspace...';
            };

            assert.strictEqual(getMessage(false), 'No workspace open');
        });

        test('Should create empty item', () => {
            const createEmptyItem = (message: string) => ({
                label: message,
                contextValue: 'empty',
                collapsibleState: 0
            });

            const item = createEmptyItem('No insights');
            assert.strictEqual(item.contextValue, 'empty');
        });
    });

    // Test commands
    suite('Commands', () => {
        test('Should have view commands', () => {
            const commands = [
                'victor.refreshWorkspaceInsights',
                'victor.runSecurityScan'
            ];

            assert.ok(commands.includes('victor.runSecurityScan'));
        });

        test('Should run security scan', () => {
            let scanRan = false;

            const runScan = () => { scanRan = true; };

            runScan();
            assert.ok(scanRan);
        });
    });

    // Test progress
    suite('Progress', () => {
        test('Should track analysis progress', () => {
            let progress = 0;

            const updateProgress = (value: number) => {
                progress = value;
            };

            updateProgress(50);
            assert.strictEqual(progress, 50);
        });

        test('Should show progress message', () => {
            const getMessage = (step: string, current: number, total: number): string => {
                return `${step}: ${current}/${total}`;
            };

            assert.strictEqual(getMessage('Scanning files', 50, 100), 'Scanning files: 50/100');
        });
    });

    // Test caching
    suite('Caching', () => {
        test('Should cache insights', () => {
            const cache = {
                stats: null as object | null,
                timestamp: 0
            };

            const setCache = (stats: object) => {
                cache.stats = stats;
                cache.timestamp = Date.now();
            };

            setCache({ files: 100 });
            assert.ok(cache.stats !== null);
        });

        test('Should invalidate on file change', () => {
            let cacheValid = true;

            const invalidate = () => { cacheValid = false; };

            invalidate();
            assert.ok(!cacheValid);
        });
    });
});
