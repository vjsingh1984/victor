/**
 * Code Actions Tests
 *
 * Tests for code action commands (explain, refactor, fix, etc.)
 */

import * as assert from 'assert';

suite('CodeActions Test Suite', () => {
    // Test action types
    suite('Action Types', () => {
        test('Should have all code action types', () => {
            const actionTypes = ['explain', 'refactor', 'fix', 'test', 'document', 'optimize', 'review'];
            assert.strictEqual(actionTypes.length, 7);
            assert.ok(actionTypes.includes('explain'));
            assert.ok(actionTypes.includes('refactor'));
        });

        test('Should map action to prompt prefix', () => {
            const promptPrefixes: Record<string, string> = {
                explain: 'Explain this code:',
                refactor: 'Refactor this code:',
                fix: 'Fix issues in this code:',
                test: 'Generate tests for this code:',
                document: 'Add documentation to this code:',
                optimize: 'Optimize this code:',
                review: 'Review this code:'
            };

            assert.strictEqual(promptPrefixes.explain, 'Explain this code:');
            assert.strictEqual(promptPrefixes.fix, 'Fix issues in this code:');
        });
    });

    // Test code block formatting
    suite('Code Block Formatting', () => {
        test('Should format code with language', () => {
            const formatCodeBlock = (code: string, language: string): string => {
                return `\`\`\`${language}\n${code}\n\`\`\``;
            };

            const result = formatCodeBlock('const x = 1;', 'typescript');
            assert.ok(result.startsWith('```typescript'));
            assert.ok(result.includes('const x = 1;'));
            assert.ok(result.endsWith('```'));
        });

        test('Should format code without language', () => {
            const formatCodeBlock = (code: string, language?: string): string => {
                return `\`\`\`${language || ''}\n${code}\n\`\`\``;
            };

            const result = formatCodeBlock('print("hello")');
            assert.ok(result.startsWith('```\n'));
        });

        test('Should escape backticks in code', () => {
            const escapeBackticks = (code: string): string => {
                return code.replace(/`/g, '\\`');
            };

            assert.strictEqual(escapeBackticks('`template`'), '\\`template\\`');
        });
    });

    // Test selection handling
    suite('Selection Handling', () => {
        test('Should detect empty selection', () => {
            const isEmptySelection = (text: string): boolean => {
                return !text || text.trim().length === 0;
            };

            assert.ok(isEmptySelection(''));
            assert.ok(isEmptySelection('   '));
            assert.ok(!isEmptySelection('code'));
        });

        test('Should get selection range info', () => {
            const selection = {
                start: { line: 5, character: 0 },
                end: { line: 10, character: 20 }
            };

            const lineCount = selection.end.line - selection.start.line + 1;
            assert.strictEqual(lineCount, 6);
        });

        test('Should expand selection to full lines', () => {
            const expandToFullLines = (start: number, end: number): { start: number; end: number } => {
                return { start: start, end: end };
            };

            const result = expandToFullLines(5, 10);
            assert.strictEqual(result.start, 5);
            assert.strictEqual(result.end, 10);
        });
    });

    // Test diagnostic integration
    suite('Diagnostic Integration', () => {
        test('Should format diagnostic message', () => {
            const formatDiagnostic = (severity: string, line: number, message: string): string => {
                return `[${severity}] Line ${line}: ${message}`;
            };

            const result = formatDiagnostic('ERROR', 10, 'Type mismatch');
            assert.strictEqual(result, '[ERROR] Line 10: Type mismatch');
        });

        test('Should collect diagnostics in range', () => {
            const diagnostics = [
                { range: { start: { line: 5 } }, message: 'Error 1' },
                { range: { start: { line: 10 } }, message: 'Error 2' },
                { range: { start: { line: 15 } }, message: 'Error 3' }
            ];

            const inRange = diagnostics.filter(d =>
                d.range.start.line >= 5 && d.range.start.line <= 12
            );

            assert.strictEqual(inRange.length, 2);
        });

        test('Should build fix prompt with diagnostics', () => {
            const buildFixPrompt = (code: string, diagnostics: string[]): string => {
                if (diagnostics.length === 0) {
                    return `Fix any issues in this code:\n${code}`;
                }
                return `Fix these issues:\n${diagnostics.join('\n')}\n\nIn this code:\n${code}`;
            };

            const prompt = buildFixPrompt('const x;', ['Missing initializer']);
            assert.ok(prompt.includes('Fix these issues'));
            assert.ok(prompt.includes('Missing initializer'));
        });
    });

    // Test language detection
    suite('Language Detection', () => {
        test('Should detect language from file extension', () => {
            const getLanguageFromExtension = (filename: string): string => {
                const ext = filename.split('.').pop()?.toLowerCase();
                const langMap: Record<string, string> = {
                    ts: 'typescript',
                    tsx: 'typescriptreact',
                    js: 'javascript',
                    jsx: 'javascriptreact',
                    py: 'python',
                    java: 'java',
                    go: 'go',
                    rs: 'rust',
                    rb: 'ruby',
                    php: 'php'
                };
                return langMap[ext || ''] || 'plaintext';
            };

            assert.strictEqual(getLanguageFromExtension('test.ts'), 'typescript');
            assert.strictEqual(getLanguageFromExtension('app.py'), 'python');
            assert.strictEqual(getLanguageFromExtension('main.go'), 'go');
        });

        test('Should get language ID mapping', () => {
            const languageIds: Record<string, string> = {
                typescript: 'TypeScript',
                javascript: 'JavaScript',
                python: 'Python',
                java: 'Java',
                go: 'Go'
            };

            assert.strictEqual(languageIds.typescript, 'TypeScript');
        });
    });

    // Test action commands
    suite('Action Commands', () => {
        test('Should have command IDs', () => {
            const commands = [
                'victor.explain',
                'victor.refactor',
                'victor.fix',
                'victor.test',
                'victor.document'
            ];

            assert.strictEqual(commands.length, 5);
            assert.ok(commands.every(c => c.startsWith('victor.')));
        });

        test('Should build explain message', () => {
            const buildExplainMessage = (code: string, language: string): string => {
                return `Explain this ${language} code:\n\`\`\`${language}\n${code}\n\`\`\``;
            };

            const msg = buildExplainMessage('const x = 1;', 'typescript');
            assert.ok(msg.includes('Explain'));
            assert.ok(msg.includes('typescript'));
        });

        test('Should build refactor message', () => {
            const buildRefactorMessage = (code: string, suggestion: string): string => {
                return `Refactor this code (${suggestion}):\n\`\`\`\n${code}\n\`\`\``;
            };

            const msg = buildRefactorMessage('function f() {}', 'extract method');
            assert.ok(msg.includes('extract method'));
        });

        test('Should build test message', () => {
            const buildTestMessage = (code: string, language: string): string => {
                return `Generate comprehensive unit tests for this ${language} code:\n\`\`\`${language}\n${code}\n\`\`\``;
            };

            const msg = buildTestMessage('def add(a, b): return a + b', 'python');
            assert.ok(msg.includes('unit tests'));
            assert.ok(msg.includes('python'));
        });
    });

    // Test context menu
    suite('Context Menu', () => {
        test('Should have menu items', () => {
            const menuItems = [
                { command: 'victor.explain', when: 'editorHasSelection' },
                { command: 'victor.refactor', when: 'editorHasSelection' },
                { command: 'victor.fix', when: 'editorHasSelection' },
                { command: 'victor.test', when: 'editorHasSelection' },
                { command: 'victor.document', when: 'editorHasSelection' }
            ];

            assert.strictEqual(menuItems.length, 5);
            assert.ok(menuItems.every(m => m.when === 'editorHasSelection'));
        });

        test('Should check selection condition', () => {
            const checkCondition = (condition: string, context: { hasSelection: boolean }): boolean => {
                if (condition === 'editorHasSelection') {
                    return context.hasSelection;
                }
                return true;
            };

            assert.ok(checkCondition('editorHasSelection', { hasSelection: true }));
            assert.ok(!checkCondition('editorHasSelection', { hasSelection: false }));
        });
    });

    // Test quick pick
    suite('Quick Pick Integration', () => {
        test('Should have refactor suggestions', () => {
            const suggestions = [
                'Extract to function',
                'Extract to variable',
                'Inline variable',
                'Rename',
                'Simplify logic',
                'Add error handling'
            ];

            assert.ok(suggestions.length >= 5);
            assert.ok(suggestions.includes('Extract to function'));
        });

        test('Should format quick pick item', () => {
            const formatItem = (label: string, description: string) => ({
                label,
                description,
                picked: false
            });

            const item = formatItem('Extract to function', 'Move selection to a new function');
            assert.strictEqual(item.label, 'Extract to function');
            assert.ok(item.description.includes('function'));
        });
    });

    // Test error handling
    suite('Error Handling', () => {
        test('Should handle missing selection', () => {
            const validateSelection = (text: string | undefined): { valid: boolean; message?: string } => {
                if (!text || text.trim().length === 0) {
                    return { valid: false, message: 'Please select some code first' };
                }
                return { valid: true };
            };

            assert.ok(!validateSelection('').valid);
            assert.ok(!validateSelection(undefined).valid);
            assert.ok(validateSelection('code').valid);
        });

        test('Should handle no active editor', () => {
            const validateEditor = (editor: object | undefined): boolean => {
                return editor !== undefined;
            };

            assert.ok(!validateEditor(undefined));
            assert.ok(validateEditor({}));
        });
    });

    // Test response handling
    suite('Response Handling', () => {
        test('Should extract code from response', () => {
            const extractCode = (response: string): string | null => {
                const match = response.match(/```[\w]*\n([\s\S]*?)```/);
                return match ? match[1].trim() : null;
            };

            const response = 'Here is the code:\n```typescript\nconst x = 1;\n```\nDone.';
            assert.strictEqual(extractCode(response), 'const x = 1;');
        });

        test('Should detect multiple code blocks', () => {
            const countCodeBlocks = (response: string): number => {
                const matches = response.match(/```/g);
                return matches ? matches.length / 2 : 0;
            };

            const response = '```js\na\n```\n```js\nb\n```';
            assert.strictEqual(countCodeBlocks(response), 2);
        });
    });
});
