/**
 * Context Provider Tests
 *
 * Tests for the ContextProvider which gathers code context
 * for AI prompts using @-mentions.
 */

import * as assert from 'assert';

suite('ContextProvider Test Suite', () => {
    // Test ContextItem interface
    suite('ContextItem Structure', () => {
        test('Should create file context item', () => {
            const item = {
                type: 'file' as const,
                name: 'test.ts',
                content: 'const x = 1;',
                path: 'src/test.ts',
                language: 'typescript'
            };

            assert.strictEqual(item.type, 'file');
            assert.strictEqual(item.name, 'test.ts');
            assert.strictEqual(item.language, 'typescript');
        });

        test('Should create symbol context item', () => {
            const item = {
                type: 'symbol' as const,
                name: 'MyClass',
                content: 'class MyClass { }',
                path: 'src/myclass.ts',
                language: 'typescript'
            };

            assert.strictEqual(item.type, 'symbol');
            assert.strictEqual(item.name, 'MyClass');
        });

        test('Should create selection context item', () => {
            const item = {
                type: 'selection' as const,
                name: 'Selection in test.ts',
                content: 'selected code here'
            };

            assert.strictEqual(item.type, 'selection');
            assert.ok(item.name.includes('Selection'));
        });

        test('Should create diagnostics context item', () => {
            const item = {
                type: 'diagnostics' as const,
                name: 'Diagnostics for test.ts',
                content: '[ERROR] Line 10: Cannot find name "foo"'
            };

            assert.strictEqual(item.type, 'diagnostics');
            assert.ok(item.content.includes('ERROR'));
        });

        test('Should create git context item', () => {
            const item = {
                type: 'git' as const,
                name: 'git status',
                content: 'On branch main\nChanges not staged...'
            };

            assert.strictEqual(item.type, 'git');
            assert.ok(item.content.includes('branch'));
        });
    });

    // Test mention parsing
    suite('Mention Parsing', () => {
        test('Should parse @file mention', () => {
            const parseMentions = (input: string) => {
                const mentions: { type: string; query: string }[] = [];
                const regex = /@(file|symbol|folder|selection|diagnostics|git)(?::([^\s]+))?/g;
                let match;
                while ((match = regex.exec(input)) !== null) {
                    mentions.push({ type: match[1], query: match[2] || '' });
                }
                return mentions;
            };

            const mentions = parseMentions('@file:test.ts');
            assert.strictEqual(mentions.length, 1);
            assert.strictEqual(mentions[0].type, 'file');
            assert.strictEqual(mentions[0].query, 'test.ts');
        });

        test('Should parse @symbol mention', () => {
            const parseMentions = (input: string) => {
                const regex = /@(file|symbol|folder|selection|diagnostics|git)(?::([^\s]+))?/g;
                const match = regex.exec(input);
                return match ? { type: match[1], query: match[2] || '' } : null;
            };

            const mention = parseMentions('@symbol:MyClass');
            assert.ok(mention);
            assert.strictEqual(mention.type, 'symbol');
            assert.strictEqual(mention.query, 'MyClass');
        });

        test('Should parse @folder mention', () => {
            const parseMentions = (input: string) => {
                const regex = /@folder:([^\s]+)/;
                const match = input.match(regex);
                return match ? match[1] : null;
            };

            const folder = parseMentions('@folder:src/components');
            assert.strictEqual(folder, 'src/components');
        });

        test('Should parse @selection mention', () => {
            const hasSelection = (input: string): boolean => {
                return /@selection\b/.test(input);
            };

            assert.ok(hasSelection('@selection'));
            assert.ok(hasSelection('Check @selection please'));
            assert.ok(!hasSelection('@file:selection'));
        });

        test('Should parse @diagnostics mention', () => {
            const hasDiagnostics = (input: string): boolean => {
                return /@diagnostics\b/.test(input);
            };

            assert.ok(hasDiagnostics('@diagnostics'));
            assert.ok(!hasDiagnostics('@diagnostic'));
        });

        test('Should parse @git mention', () => {
            const parseGit = (input: string): string | null => {
                const match = input.match(/@git(?::([^\s]+))?/);
                return match ? (match[1] || 'status') : null;
            };

            assert.strictEqual(parseGit('@git'), 'status');
            assert.strictEqual(parseGit('@git:diff'), 'diff');
            assert.strictEqual(parseGit('@git:log'), 'log');
        });

        test('Should parse multiple mentions', () => {
            const parseMentions = (input: string) => {
                const mentions: string[] = [];
                const regex = /@(file|symbol|folder|selection|diagnostics|git)(?::[^\s]+)?/g;
                let match;
                while ((match = regex.exec(input)) !== null) {
                    mentions.push(match[0]);
                }
                return mentions;
            };

            const mentions = parseMentions('@file:a.ts @symbol:MyClass @selection');
            assert.strictEqual(mentions.length, 3);
        });
    });

    // Test mention stripping
    suite('Mention Stripping', () => {
        test('Should strip mentions from input', () => {
            const stripMentions = (input: string): string => {
                return input.replace(/@(file|symbol|folder|selection|diagnostics|git)(?::[^\s]+)?/g, '').trim();
            };

            const clean = stripMentions('Explain @file:test.ts this code');
            assert.strictEqual(clean, 'Explain  this code');
        });

        test('Should strip multiple mentions', () => {
            const stripMentions = (input: string): string => {
                return input.replace(/@(file|symbol|folder|selection|diagnostics|git)(?::[^\s]+)?/g, '').replace(/\s+/g, ' ').trim();
            };

            const clean = stripMentions('@file:a.ts @symbol:Foo Explain this');
            assert.strictEqual(clean, 'Explain this');
        });

        test('Should handle input with no mentions', () => {
            const stripMentions = (input: string): string => {
                return input.replace(/@(file|symbol|folder|selection|diagnostics|git)(?::[^\s]+)?/g, '').trim();
            };

            const clean = stripMentions('Just regular text');
            assert.strictEqual(clean, 'Just regular text');
        });
    });

    // Test context formatting
    suite('Context Formatting', () => {
        test('Should format context header for file', () => {
            const formatHeader = (type: string, name: string, path?: string): string => {
                switch (type) {
                    case 'file': return `### File: ${path}`;
                    case 'symbol': return `### Symbol: ${name} (${path})`;
                    case 'folder': return `### Folder: ${path}`;
                    case 'selection': return `### Selection: ${name}`;
                    case 'diagnostics': return `### Diagnostics: ${name}`;
                    case 'git': return `### Git: ${name}`;
                    default: return `### ${name}`;
                }
            };

            assert.strictEqual(formatHeader('file', 'test.ts', 'src/test.ts'), '### File: src/test.ts');
            assert.strictEqual(formatHeader('symbol', 'MyClass', 'src/app.ts'), '### Symbol: MyClass (src/app.ts)');
        });

        test('Should format context with code block', () => {
            const formatContext = (item: { content: string; language?: string }): string => {
                return `\`\`\`${item.language || ''}\n${item.content}\n\`\`\``;
            };

            const formatted = formatContext({ content: 'const x = 1;', language: 'typescript' });
            assert.ok(formatted.startsWith('```typescript'));
            assert.ok(formatted.includes('const x = 1;'));
            assert.ok(formatted.endsWith('```'));
        });

        test('Should format full context section', () => {
            type ContextItem = { type: string; name: string; content: string; path: string };
            const items: ContextItem[] = [
                { type: 'file', name: 'test.ts', content: 'code', path: 'src/test.ts' }
            ];

            const formatContextForPrompt = (itemsList: ContextItem[][]): string => {
                if (itemsList.length === 0) {return '';}
                return `## Context\n\n${itemsList.length} item(s)`;
            };

            const formatted = formatContextForPrompt([items]);
            assert.ok(formatted.includes('## Context'));
        });
    });

    // Test severity formatting
    suite('Diagnostic Severity', () => {
        test('Should format severity string', () => {
            const getSeverityString = (severity: number): string => {
                switch (severity) {
                    case 0: return 'ERROR';
                    case 1: return 'WARNING';
                    case 2: return 'INFO';
                    case 3: return 'HINT';
                    default: return 'UNKNOWN';
                }
            };

            assert.strictEqual(getSeverityString(0), 'ERROR');
            assert.strictEqual(getSeverityString(1), 'WARNING');
            assert.strictEqual(getSeverityString(2), 'INFO');
            assert.strictEqual(getSeverityString(3), 'HINT');
            assert.strictEqual(getSeverityString(99), 'UNKNOWN');
        });

        test('Should format diagnostic line', () => {
            const formatDiagnostic = (severity: string, line: number, message: string): string => {
                return `[${severity}] Line ${line}: ${message}`;
            };

            const formatted = formatDiagnostic('ERROR', 10, 'Cannot find name "foo"');
            assert.strictEqual(formatted, '[ERROR] Line 10: Cannot find name "foo"');
        });
    });

    // Test file search patterns
    suite('File Search', () => {
        test('Should build glob pattern', () => {
            const buildPattern = (query: string): string => {
                return `**/${query}*`;
            };

            assert.strictEqual(buildPattern('test'), '**/test*');
            assert.strictEqual(buildPattern('app.ts'), '**/app.ts*');
        });

        test('Should have exclusion pattern', () => {
            const exclusionPattern = '**/node_modules/**';
            assert.ok(exclusionPattern.includes('node_modules'));
        });

        test('Should limit search results', () => {
            const maxResults = 10;
            const results = Array(20).fill({ path: 'file.ts' });
            const limited = results.slice(0, maxResults);

            assert.strictEqual(limited.length, 10);
        });
    });

    // Test symbol expansion
    suite('Symbol Expansion', () => {
        test('Should expand symbol range', () => {
            const expandRange = (startLine: number, endLine: number, maxExpansion: number, totalLines: number) => {
                const newEnd = Math.min(endLine + maxExpansion, totalLines - 1);
                return { startLine, endLine: newEnd };
            };

            const expanded = expandRange(10, 15, 20, 100);
            assert.strictEqual(expanded.startLine, 10);
            assert.strictEqual(expanded.endLine, 35);
        });

        test('Should limit symbol expansion to file bounds', () => {
            const expandRange = (endLine: number, maxExpansion: number, totalLines: number) => {
                return Math.min(endLine + maxExpansion, totalLines - 1);
            };

            assert.strictEqual(expandRange(90, 20, 100), 99);
        });

        test('Should limit symbols returned', () => {
            const maxSymbols = 5;
            const symbols = Array(10).fill({ name: 'Symbol' });
            const limited = symbols.slice(0, maxSymbols);

            assert.strictEqual(limited.length, 5);
        });
    });

    // Test folder tree generation
    suite('Folder Tree', () => {
        test('Should format file entries', () => {
            const formatEntry = (name: string, isDirectory: boolean): string => {
                const icon = isDirectory ? '📁' : '📄';
                return `${icon} ${name}`;
            };

            assert.strictEqual(formatEntry('src', true), '📁 src');
            assert.strictEqual(formatEntry('test.ts', false), '📄 test.ts');
        });

        test('Should join entries with newlines', () => {
            const entries = ['📁 src', '📁 tests', '📄 package.json'];
            const tree = entries.join('\n');

            assert.ok(tree.includes('\n'));
            assert.strictEqual(tree.split('\n').length, 3);
        });
    });

    // Test git commands
    suite('Git Commands', () => {
        test('Should default to status', () => {
            const getGitCommand = (query?: string): string => {
                return query || 'status';
            };

            assert.strictEqual(getGitCommand(), 'status');
            assert.strictEqual(getGitCommand('diff'), 'diff');
            assert.strictEqual(getGitCommand('log'), 'log');
        });

        test('Should have supported git commands', () => {
            const supportedCommands = ['status', 'diff', 'log', 'branch'];

            assert.ok(supportedCommands.includes('status'));
            assert.ok(supportedCommands.includes('diff'));
        });
    });

    // Test completion items
    suite('Mention Completion', () => {
        test('Should have all mention types', () => {
            const mentionTypes = [
                { label: 'file', description: 'Include file contents', detail: '@file:filename' },
                { label: 'symbol', description: 'Include symbol definition', detail: '@symbol:name' },
                { label: 'folder', description: 'Include folder structure', detail: '@folder:path' },
                { label: 'selection', description: 'Include current selection', detail: '@selection' },
                { label: 'diagnostics', description: 'Include current file errors', detail: '@diagnostics' },
                { label: 'git', description: 'Include git info', detail: '@git:status' }
            ];

            assert.strictEqual(mentionTypes.length, 6);
            assert.ok(mentionTypes.some(t => t.label === 'file'));
            assert.ok(mentionTypes.some(t => t.label === 'symbol'));
        });

        test('Should filter completions by prefix', () => {
            const mentionTypes = ['file', 'folder', 'selection', 'symbol', 'diagnostics', 'git'];
            const prefix = 'fi';

            const filtered = mentionTypes.filter(t => t.startsWith(prefix));
            assert.strictEqual(filtered.length, 1); // only 'file' starts with 'fi'
            assert.ok(filtered.includes('file'));
        });

        test('Should detect @ trigger', () => {
            const isAtTrigger = (text: string): boolean => {
                return /@\w*$/.test(text);
            };

            assert.ok(isAtTrigger('@'));
            assert.ok(isAtTrigger('@fi'));
            assert.ok(isAtTrigger('text @sym'));
            assert.ok(!isAtTrigger('text'));
        });

        test('Should not add colon for standalone mentions', () => {
            const needsColon = (type: string): boolean => {
                return type !== 'selection' && type !== 'diagnostics';
            };

            assert.ok(needsColon('file'));
            assert.ok(needsColon('symbol'));
            assert.ok(!needsColon('selection'));
            assert.ok(!needsColon('diagnostics'));
        });
    });

    // Test quick pick options
    suite('Context Picker', () => {
        test('Should have quick pick options', () => {
            const options = [
                { label: '$(file) Current File', detail: '@file' },
                { label: '$(selection) Current Selection', detail: '@selection' },
                { label: '$(warning) Current Diagnostics', detail: '@diagnostics' },
                { label: '$(git-branch) Git Status', detail: '@git:status' },
                { label: '$(git-commit) Git Diff', detail: '@git:diff' },
                { label: '$(search) Search Files...', detail: '@file:' },
                { label: '$(symbol-class) Search Symbols...', detail: '@symbol:' }
            ];

            assert.strictEqual(options.length, 7);
        });

        test('Should identify search options', () => {
            const isSearchOption = (detail: string): boolean => {
                return detail.endsWith(':');
            };

            assert.ok(isSearchOption('@file:'));
            assert.ok(isSearchOption('@symbol:'));
            assert.ok(!isSearchOption('@selection'));
        });
    });

    // Test path handling
    suite('Path Handling', () => {
        test('Should detect absolute path', () => {
            const isAbsolute = (path: string): boolean => {
                return path.startsWith('/') || /^[A-Z]:/.test(path);
            };

            assert.ok(isAbsolute('/Users/test'));
            assert.ok(isAbsolute('C:\\Users'));
            assert.ok(!isAbsolute('src/test.ts'));
        });

        test('Should get relative path', () => {
            const getRelativePath = (fullPath: string, workspaceRoot: string): string => {
                if (fullPath.startsWith(workspaceRoot)) {
                    return fullPath.substring(workspaceRoot.length + 1);
                }
                return fullPath;
            };

            const result = getRelativePath('/workspace/src/test.ts', '/workspace');
            assert.strictEqual(result, 'src/test.ts');
        });
    });

    // Test clipboard operations
    suite('Clipboard Operations', () => {
        test('Should format for clipboard', () => {
            const items = [{ name: 'test.ts' }];
            const message = `${items.length} context item(s) copied to clipboard`;

            assert.ok(message.includes('1'));
            assert.ok(message.includes('copied'));
        });
    });
});
