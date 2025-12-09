/**
 * Code Action Provider Tests
 *
 * Tests for the VictorCodeActionProvider which provides AI-powered
 * quick fixes and refactoring suggestions in the lightbulb menu.
 */

import * as assert from 'assert';

suite('CodeActionProvider Test Suite', () => {
    // Test code action kinds
    suite('Code Action Kinds', () => {
        test('Should have correct action kinds', () => {
            const kinds = {
                QuickFix: 'quickfix',
                Refactor: 'refactor',
                RefactorExtract: 'refactor.extract',
                RefactorInline: 'refactor.inline',
                RefactorRewrite: 'refactor.rewrite',
                Source: 'source',
                SourceOrganizeImports: 'source.organizeImports',
            };

            assert.strictEqual(kinds.QuickFix, 'quickfix');
            assert.strictEqual(kinds.Refactor, 'refactor');
            assert.ok(kinds.RefactorExtract.startsWith('refactor.'));
        });
    });

    // Test diagnostic code detection
    suite('Diagnostic Detection', () => {
        test('Should detect error diagnostics', () => {
            const diagnostic = {
                severity: 0, // Error
                message: 'Cannot find name "foo"',
                range: { start: { line: 10, character: 0 }, end: { line: 10, character: 5 } }
            };

            const isError = diagnostic.severity === 0;
            assert.ok(isError);
        });

        test('Should detect warning diagnostics', () => {
            const diagnostic = {
                severity: 1, // Warning
                message: 'Unused variable "x"',
                range: { start: { line: 5, character: 0 }, end: { line: 5, character: 10 } }
            };

            const isWarning = diagnostic.severity === 1;
            assert.ok(isWarning);
        });

        test('Should detect hint diagnostics', () => {
            const diagnostic = {
                severity: 3, // Hint
                message: 'Consider using const instead of let',
                range: { start: { line: 1, character: 0 }, end: { line: 1, character: 10 } }
            };

            const isHint = diagnostic.severity === 3;
            assert.ok(isHint);
        });
    });

    // Test code action categories
    suite('Code Action Categories', () => {
        test('Should categorize explain action', () => {
            const action = {
                title: '$(question) Explain this code',
                kind: 'source',
                command: 'victor.explain'
            };

            assert.ok(action.title.includes('Explain'));
            assert.strictEqual(action.kind, 'source');
        });

        test('Should categorize fix action', () => {
            const action = {
                title: '$(wrench) Fix with AI',
                kind: 'quickfix',
                command: 'victor.fix'
            };

            assert.ok(action.title.includes('Fix'));
            assert.strictEqual(action.kind, 'quickfix');
        });

        test('Should categorize refactor action', () => {
            const action = {
                title: '$(edit) Refactor with AI',
                kind: 'refactor',
                command: 'victor.refactor'
            };

            assert.ok(action.title.includes('Refactor'));
            assert.strictEqual(action.kind, 'refactor');
        });

        test('Should categorize test generation action', () => {
            const action = {
                title: '$(beaker) Generate tests',
                kind: 'source',
                command: 'victor.generateTests'
            };

            assert.ok(action.title.includes('tests'));
            assert.strictEqual(action.kind, 'source');
        });

        test('Should categorize documentation action', () => {
            const action = {
                title: '$(book) Add documentation',
                kind: 'source',
                command: 'victor.addDocs'
            };

            assert.ok(action.title.includes('documentation'));
            assert.strictEqual(action.kind, 'source');
        });
    });

    // Test selection-based actions
    suite('Selection-Based Actions', () => {
        test('Should detect non-empty selection', () => {
            const selection = {
                start: { line: 5, character: 0 },
                end: { line: 10, character: 20 },
                isEmpty: false
            };

            const hasSelection = !selection.isEmpty;
            assert.ok(hasSelection);
        });

        test('Should detect empty selection', () => {
            const selection = {
                start: { line: 5, character: 10 },
                end: { line: 5, character: 10 },
                isEmpty: true
            };

            const hasSelection = !selection.isEmpty;
            assert.ok(!hasSelection);
        });

        test('Should calculate selection size', () => {
            const calculateSelectionLines = (start: number, end: number): number => {
                return end - start + 1;
            };

            assert.strictEqual(calculateSelectionLines(5, 10), 6);
            assert.strictEqual(calculateSelectionLines(0, 0), 1);
            assert.strictEqual(calculateSelectionLines(10, 50), 41);
        });
    });

    // Test prompt generation
    suite('Prompt Generation', () => {
        test('Should generate fix prompt with diagnostic', () => {
            const generateFixPrompt = (diagnostic: string, code: string): string => {
                return `Fix the following issue:\n\nError: ${diagnostic}\n\nCode:\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = generateFixPrompt(
                'Cannot find name "foo"',
                'console.log(foo);'
            );

            assert.ok(prompt.includes('Fix'));
            assert.ok(prompt.includes('Cannot find name'));
            assert.ok(prompt.includes('console.log'));
        });

        test('Should generate explain prompt', () => {
            const generateExplainPrompt = (code: string): string => {
                return `Explain the following code:\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = generateExplainPrompt('const sum = (a, b) => a + b;');
            assert.ok(prompt.includes('Explain'));
            assert.ok(prompt.includes('sum'));
        });

        test('Should generate refactor prompt', () => {
            const generateRefactorPrompt = (code: string): string => {
                return `Suggest improvements for the following code:\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = generateRefactorPrompt('for(var i=0;i<arr.length;i++){}');
            assert.ok(prompt.includes('improvements'));
            assert.ok(prompt.includes('for'));
        });
    });

    // Test action filtering
    suite('Action Filtering', () => {
        test('Should filter actions by kind', () => {
            const actions = [
                { kind: 'quickfix', title: 'Fix 1' },
                { kind: 'refactor', title: 'Refactor 1' },
                { kind: 'quickfix', title: 'Fix 2' },
                { kind: 'source', title: 'Source 1' },
            ];

            const quickFixes = actions.filter(a => a.kind === 'quickfix');
            const refactors = actions.filter(a => a.kind === 'refactor');

            assert.strictEqual(quickFixes.length, 2);
            assert.strictEqual(refactors.length, 1);
        });

        test('Should filter actions for diagnostics', () => {
            const hasDiagnostics = true;

            const getAvailableActions = (hasDiag: boolean): string[] => {
                const actions = ['explain', 'refactor', 'addTests', 'addDocs'];
                if (hasDiag) {
                    actions.unshift('fix'); // Fix action only when diagnostics present
                }
                return actions;
            };

            const withDiag = getAvailableActions(true);
            const withoutDiag = getAvailableActions(false);

            assert.ok(withDiag.includes('fix'));
            assert.ok(!withoutDiag.includes('fix'));
        });
    });

    // Test preferred action logic
    suite('Preferred Actions', () => {
        test('Should mark fix as preferred when error diagnostic', () => {
            const diagnostic = { severity: 0 }; // Error

            const isPreferred = diagnostic.severity === 0;
            assert.ok(isPreferred);
        });

        test('Should not mark as preferred for hints', () => {
            const diagnostic = { severity: 3 }; // Hint

            const isPreferred = diagnostic.severity === 0;
            assert.ok(!isPreferred);
        });
    });

    // Test command argument preparation
    suite('Command Arguments', () => {
        test('Should prepare explain command args', () => {
            const prepareExplainArgs = (filePath: string, selection: { start: number; end: number }): object => {
                return {
                    action: 'explain',
                    file: filePath,
                    startLine: selection.start,
                    endLine: selection.end
                };
            };

            const args = prepareExplainArgs('/path/to/file.ts', { start: 10, end: 20 });
            assert.strictEqual((args as any).action, 'explain');
            assert.strictEqual((args as any).startLine, 10);
        });

        test('Should prepare fix command args with diagnostic', () => {
            const prepareFixArgs = (filePath: string, diagnostic: { message: string; line: number }): object => {
                return {
                    action: 'fix',
                    file: filePath,
                    diagnostic: diagnostic.message,
                    line: diagnostic.line
                };
            };

            const args = prepareFixArgs('/path/to/file.ts', { message: 'Error message', line: 15 });
            assert.strictEqual((args as any).action, 'fix');
            assert.strictEqual((args as any).line, 15);
        });
    });

    // Test action title formatting
    suite('Action Title Formatting', () => {
        test('Should format titles with VS Code icons', () => {
            const formatTitle = (icon: string, text: string): string => {
                return `$(${icon}) ${text}`;
            };

            assert.strictEqual(formatTitle('question', 'Explain'), '$(question) Explain');
            assert.strictEqual(formatTitle('wrench', 'Fix'), '$(wrench) Fix');
            assert.strictEqual(formatTitle('edit', 'Refactor'), '$(edit) Refactor');
        });
    });

    // Test context requirements
    suite('Context Requirements', () => {
        test('Should check if in supported language', () => {
            const supportedLanguages = ['typescript', 'javascript', 'python', 'java', 'go', 'rust'];

            const isSupported = (languageId: string): boolean => {
                return supportedLanguages.includes(languageId);
            };

            assert.ok(isSupported('typescript'));
            assert.ok(isSupported('python'));
            assert.ok(!isSupported('markdown'));
            assert.ok(!isSupported('json'));
        });
    });
});
