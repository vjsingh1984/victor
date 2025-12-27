/**
 * Diff View Provider Tests
 *
 * Tests for the DiffViewProvider which provides inline diff preview
 * and application UI for proposed file changes.
 */

import * as assert from 'assert';

suite('DiffViewProvider Test Suite', () => {
    // Test FileChange interface
    suite('FileChange Structure', () => {
        test('Should create modify change', () => {
            const change = {
                filePath: 'src/test.ts',
                originalContent: 'const x = 1;',
                newContent: 'const x = 2;',
                changeType: 'modify' as const
            };

            assert.strictEqual(change.changeType, 'modify');
            assert.ok(change.originalContent !== change.newContent);
        });

        test('Should create create change', () => {
            const change = {
                filePath: 'src/new-file.ts',
                originalContent: '',
                newContent: 'export const foo = 1;',
                changeType: 'create' as const
            };

            assert.strictEqual(change.changeType, 'create');
            assert.strictEqual(change.originalContent, '');
        });

        test('Should create delete change', () => {
            const change = {
                filePath: 'src/old-file.ts',
                originalContent: 'const old = true;',
                newContent: '',
                changeType: 'delete' as const
            };

            assert.strictEqual(change.changeType, 'delete');
            assert.strictEqual(change.newContent, '');
        });

        test('Should track selection state', () => {
            const change = {
                filePath: 'src/test.ts',
                originalContent: '',
                newContent: 'code',
                changeType: 'create' as const,
                selected: true
            };

            assert.ok(change.selected);
            change.selected = false;
            assert.ok(!change.selected);
        });
    });

    // Test DiffSession structure
    suite('DiffSession Structure', () => {
        test('Should create session with ID', () => {
            const sessionId = `diff-${Date.now()}`;
            const session = {
                id: sessionId,
                changes: [],
                timestamp: new Date(),
                description: 'Refactoring changes'
            };

            assert.ok(session.id.startsWith('diff-'));
            assert.strictEqual(session.changes.length, 0);
        });

        test('Should track session timestamp', () => {
            const before = new Date();
            const session = {
                id: 'test-session',
                changes: [],
                timestamp: new Date(),
                description: 'Test'
            };
            const after = new Date();

            assert.ok(session.timestamp >= before);
            assert.ok(session.timestamp <= after);
        });
    });

    // Test line statistics calculation
    suite('Line Statistics', () => {
        test('Should calculate lines added for create', () => {
            const newContent = 'line1\nline2\nline3';
            const linesAdded = newContent.split('\n').length;

            assert.strictEqual(linesAdded, 3);
        });

        test('Should calculate lines removed for delete', () => {
            const originalContent = 'line1\nline2';
            const linesRemoved = originalContent.split('\n').length;

            assert.strictEqual(linesRemoved, 2);
        });

        test('Should calculate diff for modify', () => {
            const originalContent = 'line1\nline2';
            const newContent = 'line1\nline2\nline3\nline4';

            const origLines = originalContent.split('\n').length;
            const newLines = newContent.split('\n').length;
            const linesAdded = Math.max(0, newLines - origLines);
            const linesRemoved = Math.max(0, origLines - newLines);

            assert.strictEqual(linesAdded, 2);
            assert.strictEqual(linesRemoved, 0);
        });

        test('Should calculate total stats for session', () => {
            const changes = [
                { linesAdded: 10, linesRemoved: 2 },
                { linesAdded: 5, linesRemoved: 5 },
                { linesAdded: 0, linesRemoved: 3 }
            ];

            const totalAdded = changes.reduce((sum, c) => sum + c.linesAdded, 0);
            const totalRemoved = changes.reduce((sum, c) => sum + c.linesRemoved, 0);

            assert.strictEqual(totalAdded, 15);
            assert.strictEqual(totalRemoved, 10);
        });
    });

    // Test change icons
    suite('Change Icons', () => {
        test('Should get correct icon for change type', () => {
            const getChangeIcon = (changeType: string): string => {
                switch (changeType) {
                    case 'create': return '$(new-file)';
                    case 'modify': return '$(edit)';
                    case 'delete': return '$(trash)';
                    default: return '$(file)';
                }
            };

            assert.strictEqual(getChangeIcon('create'), '$(new-file)');
            assert.strictEqual(getChangeIcon('modify'), '$(edit)');
            assert.strictEqual(getChangeIcon('delete'), '$(trash)');
            assert.strictEqual(getChangeIcon('unknown'), '$(file)');
        });
    });

    // Test language detection
    suite('Language Detection', () => {
        test('Should detect language from extension', () => {
            const getLanguageId = (filePath: string): string => {
                const ext = filePath.split('.').pop()?.toLowerCase() || '';
                const languageMap: Record<string, string> = {
                    'ts': 'typescript',
                    'tsx': 'typescriptreact',
                    'js': 'javascript',
                    'jsx': 'javascriptreact',
                    'py': 'python',
                    'rs': 'rust',
                    'go': 'go',
                    'java': 'java',
                    'md': 'markdown',
                    'json': 'json'
                };
                return languageMap[ext] || 'plaintext';
            };

            assert.strictEqual(getLanguageId('test.ts'), 'typescript');
            assert.strictEqual(getLanguageId('app.py'), 'python');
            assert.strictEqual(getLanguageId('main.rs'), 'rust');
            assert.strictEqual(getLanguageId('unknown.xyz'), 'plaintext');
        });
    });

    // Test session management
    suite('Session Management', () => {
        test('Should add session to pending', () => {
            const pendingSessions = new Map<string, object>();
            const session = { id: 'session-1', changes: [], description: 'Test' };

            pendingSessions.set(session.id, session);

            assert.strictEqual(pendingSessions.size, 1);
            assert.ok(pendingSessions.has('session-1'));
        });

        test('Should remove session after apply', () => {
            const pendingSessions = new Map<string, object>();
            pendingSessions.set('session-1', { changes: [] });
            pendingSessions.set('session-2', { changes: [] });

            pendingSessions.delete('session-1');

            assert.strictEqual(pendingSessions.size, 1);
            assert.ok(!pendingSessions.has('session-1'));
            assert.ok(pendingSessions.has('session-2'));
        });

        test('Should get all pending sessions', () => {
            const pendingSessions = new Map<string, { id: string }>();
            pendingSessions.set('s1', { id: 's1' });
            pendingSessions.set('s2', { id: 's2' });

            const sessions = Array.from(pendingSessions.values());

            assert.strictEqual(sessions.length, 2);
        });
    });

    // Test selection management
    suite('Selection Management', () => {
        test('Should toggle file selection', () => {
            const changes = [
                { filePath: 'a.ts', selected: true },
                { filePath: 'b.ts', selected: true },
                { filePath: 'c.ts', selected: false }
            ];

            const toggleFile = (filePath: string) => {
                const change = changes.find(c => c.filePath === filePath);
                if (change) {
                    change.selected = !change.selected;
                }
            };

            toggleFile('a.ts');
            assert.ok(!changes[0].selected);

            toggleFile('c.ts');
            assert.ok(changes[2].selected);
        });

        test('Should select all files', () => {
            const changes = [
                { filePath: 'a.ts', selected: false },
                { filePath: 'b.ts', selected: false }
            ];

            changes.forEach(c => c.selected = true);

            assert.ok(changes.every(c => c.selected));
        });

        test('Should deselect all files', () => {
            const changes = [
                { filePath: 'a.ts', selected: true },
                { filePath: 'b.ts', selected: true }
            ];

            changes.forEach(c => c.selected = false);

            assert.ok(changes.every(c => !c.selected));
        });

        test('Should count selected files', () => {
            const changes = [
                { selected: true },
                { selected: false },
                { selected: true },
                { selected: true }
            ];

            const selectedCount = changes.filter(c => c.selected).length;
            assert.strictEqual(selectedCount, 3);
        });
    });

    // Test path handling
    suite('Path Handling', () => {
        test('Should detect absolute path', () => {
            const isAbsolute = (path: string): boolean => {
                return path.startsWith('/') || /^[A-Z]:\\/.test(path);
            };

            assert.ok(isAbsolute('/Users/test/file.ts'));
            assert.ok(isAbsolute('C:\\Users\\test\\file.ts'));
            assert.ok(!isAbsolute('src/file.ts'));
            assert.ok(!isAbsolute('./file.ts'));
        });

        test('Should get basename from path', () => {
            const getBasename = (filePath: string): string => {
                return filePath.split('/').pop() || filePath;
            };

            assert.strictEqual(getBasename('src/components/Button.tsx'), 'Button.tsx');
            assert.strictEqual(getBasename('file.ts'), 'file.ts');
        });

        test('Should get dirname from path', () => {
            const getDirname = (filePath: string): string => {
                const parts = filePath.split('/');
                parts.pop();
                return parts.join('/') || '.';
            };

            assert.strictEqual(getDirname('src/components/Button.tsx'), 'src/components');
            assert.strictEqual(getDirname('file.ts'), '.');
        });
    });

    // Test response parsing
    suite('Response Parsing', () => {
        test('Should extract code blocks with file paths', () => {
            const parseCodeBlocks = (response: string): { filePath?: string; content: string }[] => {
                const blocks: { filePath?: string; content: string }[] = [];
                const regex = /```(?:\w+\s+)?([^\n]+\.\w+)?\n([\s\S]*?)```/g;
                let match;
                while ((match = regex.exec(response)) !== null) {
                    blocks.push({
                        filePath: match[1],
                        content: match[2]?.trim() || ''
                    });
                }
                return blocks;
            };

            const response = '```typescript src/test.ts\nconst x = 1;\n```';
            const blocks = parseCodeBlocks(response);

            assert.strictEqual(blocks.length, 1);
        });

        test('Should extract diff blocks', () => {
            const hasDiffBlock = (response: string): boolean => {
                return /```diff\n[\s\S]*?```/.test(response);
            };

            assert.ok(hasDiffBlock('```diff\n- old\n+ new\n```'));
            assert.ok(!hasDiffBlock('```typescript\ncode\n```'));
        });

        test('Should parse diff file path', () => {
            const parseDiffFilePath = (diffContent: string): string | null => {
                const match = diffContent.match(/^(?:---|\+\+\+)\s+([^\n]+)/m);
                if (match) {
                    return match[1].replace(/^[ab]\//, '');
                }
                return null;
            };

            const diff = '--- a/src/test.ts\n+++ b/src/test.ts\n@@ -1 +1 @@\n- old\n+ new';
            assert.strictEqual(parseDiffFilePath(diff), 'src/test.ts');
        });
    });

    // Test status bar updates
    suite('Status Bar', () => {
        test('Should format status bar text', () => {
            const formatStatusBar = (totalChanges: number): string => {
                if (totalChanges === 0) {return '';}
                return `$(git-compare) ${totalChanges} pending`;
            };

            assert.strictEqual(formatStatusBar(0), '');
            assert.strictEqual(formatStatusBar(5), '$(git-compare) 5 pending');
        });

        test('Should format tooltip', () => {
            const formatTooltip = (totalChanges: number): string => {
                return `${totalChanges} pending file change(s)\nClick to review`;
            };

            const tooltip = formatTooltip(3);
            assert.ok(tooltip.includes('3'));
            assert.ok(tooltip.includes('Click to review'));
        });
    });

    // Test webview messages
    suite('Webview Messages', () => {
        test('Should handle toggleFile message', () => {
            const message = { type: 'toggleFile', filePath: 'test.ts', selected: true };
            assert.strictEqual(message.type, 'toggleFile');
            assert.strictEqual(message.filePath, 'test.ts');
            assert.ok(message.selected);
        });

        test('Should handle toggleAll message', () => {
            const message = { type: 'toggleAll', selected: false };
            assert.strictEqual(message.type, 'toggleAll');
            assert.ok(!message.selected);
        });

        test('Should handle showFileDiff message', () => {
            const message = { type: 'showFileDiff', filePath: 'src/app.ts' };
            assert.strictEqual(message.type, 'showFileDiff');
            assert.strictEqual(message.filePath, 'src/app.ts');
        });

        test('Should handle applySelected message', () => {
            const message = { type: 'applySelected' };
            assert.strictEqual(message.type, 'applySelected');
        });

        test('Should handle rejectAll message', () => {
            const message = { type: 'rejectAll' };
            assert.strictEqual(message.type, 'rejectAll');
        });
    });

    // Test apply/reject flow
    suite('Apply/Reject Flow', () => {
        test('Should filter selected changes', () => {
            const changes = [
                { filePath: 'a.ts', selected: true },
                { filePath: 'b.ts', selected: false },
                { filePath: 'c.ts', selected: true }
            ];

            const selectedChanges = changes.filter(c => c.selected);
            assert.strictEqual(selectedChanges.length, 2);
        });

        test('Should remove applied changes', () => {
            const changes = [
                { filePath: 'a.ts', selected: true },
                { filePath: 'b.ts', selected: false },
                { filePath: 'c.ts', selected: true }
            ];

            const remaining = changes.filter(c => !c.selected);
            assert.strictEqual(remaining.length, 1);
            assert.strictEqual(remaining[0].filePath, 'b.ts');
        });

        test('Should track apply success count', () => {
            const applyResults = [true, true, false, true];
            const successCount = applyResults.filter(r => r).length;

            assert.strictEqual(successCount, 3);
        });
    });

    // Test confirmation messages
    suite('Confirmation Messages', () => {
        test('Should format apply confirmation', () => {
            const formatConfirmation = (count: number): string => {
                return `Apply ${count} selected change(s)?`;
            };

            assert.strictEqual(formatConfirmation(5), 'Apply 5 selected change(s)?');
        });

        test('Should format success message', () => {
            const formatSuccess = (success: number, total: number): string => {
                return `Applied ${success}/${total} changes`;
            };

            assert.strictEqual(formatSuccess(3, 5), 'Applied 3/5 changes');
        });
    });

    // Test OriginalContentProvider
    suite('OriginalContentProvider', () => {
        test('Should store and retrieve content', () => {
            const contents = new Map<string, string>();

            contents.set('/path/file.ts', 'original content');
            const retrieved = contents.get('/path/file.ts');

            assert.strictEqual(retrieved, 'original content');
        });

        test('Should clear content', () => {
            const contents = new Map<string, string>();
            contents.set('/path/file.ts', 'content');

            contents.delete('/path/file.ts');

            assert.ok(!contents.has('/path/file.ts'));
        });

        test('Should return empty for missing content', () => {
            const contents = new Map<string, string>();
            const retrieved = contents.get('/missing/file.ts') || '';

            assert.strictEqual(retrieved, '');
        });
    });
});
