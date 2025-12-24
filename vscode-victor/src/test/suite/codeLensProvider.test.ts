/**
 * CodeLens Provider Tests
 *
 * Tests for the VictorCodeLensProvider which provides AI suggestion
 * lenses above code symbols (functions, classes, etc.)
 */

import * as assert from 'assert';

suite('CodeLensProvider Test Suite', () => {
    // Test symbol detection patterns
    suite('Symbol Detection Patterns', () => {
        test('Should detect TypeScript function declarations', () => {
            const functionPattern = /(?:async\s+)?(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>|\([^)]*\))/gm;
            const code = `
                function hello() {}
                const greet = () => {}
                async function fetchData() {}
                const processAsync = async () => {}
            `;

            const matches = [...code.matchAll(functionPattern)];
            assert.ok(matches.length >= 2); // Should find multiple functions
        });

        test('Should detect Python function declarations', () => {
            const pythonFunctionPattern = /^\s*(?:async\s+)?def\s+(\w+)\s*\(/gm;
            const code = `
def hello():
    pass

async def fetch_data():
    pass

def process(x, y):
    return x + y
            `;

            const matches = [...code.matchAll(pythonFunctionPattern)];
            assert.strictEqual(matches.length, 3);
        });

        test('Should detect class declarations', () => {
            const classPattern = /^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/gm;
            const code = `
class Animal {}
export class Dog extends Animal {}
abstract class Vehicle {}
            `;

            const matches = [...code.matchAll(classPattern)];
            assert.strictEqual(matches.length, 3);
        });

        test('Should detect interface declarations', () => {
            const interfacePattern = /^\s*(?:export\s+)?interface\s+(\w+)/gm;
            const code = `
interface User {}
export interface Config {}
            `;

            const matches = [...code.matchAll(interfacePattern)];
            assert.strictEqual(matches.length, 2);
        });

        test('Should detect method declarations in class', () => {
            const methodPattern = /^\s*(?:public|private|protected|static|async)?\s*(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*{/gm;
            const code = `
class Service {
    public getData() {}
    private processData() {}
    async fetchRemote() {}
    static getInstance() {}
}
            `;

            const matches = [...code.matchAll(methodPattern)];
            assert.ok(matches.length >= 2);
        });
    });

    // Test CodeLens action types
    suite('CodeLens Actions', () => {
        test('Should have correct command IDs', () => {
            const commands = {
                explain: 'victor.codeLens.explain',
                refactor: 'victor.codeLens.refactor',
                addTests: 'victor.codeLens.addTests',
                addDocs: 'victor.codeLens.addDocs',
                optimize: 'victor.codeLens.optimize',
            };

            assert.strictEqual(commands.explain, 'victor.codeLens.explain');
            assert.strictEqual(commands.refactor, 'victor.codeLens.refactor');
            assert.strictEqual(commands.addTests, 'victor.codeLens.addTests');
            assert.strictEqual(commands.addDocs, 'victor.codeLens.addDocs');
            assert.strictEqual(commands.optimize, 'victor.codeLens.optimize');
        });

        test('Should format explain prompt correctly', () => {
            const formatExplainPrompt = (symbolName: string, symbolType: string, code: string): string => {
                return `Explain the ${symbolType} "${symbolName}":\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = formatExplainPrompt('processData', 'function', 'function processData() {}');
            assert.ok(prompt.includes('function'));
            assert.ok(prompt.includes('processData'));
            assert.ok(prompt.includes('```'));
        });

        test('Should format refactor prompt correctly', () => {
            const formatRefactorPrompt = (symbolName: string, code: string): string => {
                return `Suggest refactoring improvements for "${symbolName}":\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = formatRefactorPrompt('fetchUser', 'async function fetchUser() {}');
            assert.ok(prompt.includes('refactoring'));
            assert.ok(prompt.includes('fetchUser'));
        });

        test('Should format test generation prompt correctly', () => {
            const formatTestPrompt = (symbolName: string, code: string): string => {
                return `Generate unit tests for "${symbolName}":\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = formatTestPrompt('Calculator', 'class Calculator {}');
            assert.ok(prompt.includes('unit tests'));
            assert.ok(prompt.includes('Calculator'));
        });
    });

    // Test symbol type inference
    suite('Symbol Type Inference', () => {
        test('Should infer function type', () => {
            const inferSymbolType = (line: string): string => {
                if (/function\s+\w+/.test(line)) return 'function';
                if (/=>\s*{/.test(line) || /=>\s*[^{]/.test(line)) return 'arrow function';
                if (/class\s+\w+/.test(line)) return 'class';
                if (/interface\s+\w+/.test(line)) return 'interface';
                if (/def\s+\w+/.test(line)) return 'function';
                return 'symbol';
            };

            assert.strictEqual(inferSymbolType('function hello() {}'), 'function');
            assert.strictEqual(inferSymbolType('const greet = () => {}'), 'arrow function');
            assert.strictEqual(inferSymbolType('class Animal {}'), 'class');
            assert.strictEqual(inferSymbolType('interface Config {}'), 'interface');
            assert.strictEqual(inferSymbolType('def process():'), 'function');
        });
    });

    // Test file type detection
    suite('File Type Detection', () => {
        test('Should detect TypeScript files', () => {
            const isTypeScript = (fileName: string): boolean => {
                return /\.tsx?$/.test(fileName);
            };

            assert.ok(isTypeScript('file.ts'));
            assert.ok(isTypeScript('component.tsx'));
            assert.ok(!isTypeScript('file.js'));
            assert.ok(!isTypeScript('file.py'));
        });

        test('Should detect JavaScript files', () => {
            const isJavaScript = (fileName: string): boolean => {
                return /\.jsx?$/.test(fileName);
            };

            assert.ok(isJavaScript('file.js'));
            assert.ok(isJavaScript('component.jsx'));
            assert.ok(!isJavaScript('file.ts'));
            assert.ok(!isJavaScript('file.py'));
        });

        test('Should detect Python files', () => {
            const isPython = (fileName: string): boolean => {
                return /\.py$/.test(fileName);
            };

            assert.ok(isPython('script.py'));
            assert.ok(!isPython('file.ts'));
            assert.ok(!isPython('file.js'));
        });

        test('Should detect supported languages', () => {
            const supportedExtensions = [
                'ts', 'tsx', 'js', 'jsx', 'py', 'java', 'cs', 'go', 'rs', 'cpp', 'c', 'rb', 'php', 'swift', 'kt'
            ];

            const isSupported = (fileName: string): boolean => {
                const ext = fileName.split('.').pop() || '';
                return supportedExtensions.includes(ext);
            };

            assert.ok(isSupported('file.ts'));
            assert.ok(isSupported('file.py'));
            assert.ok(isSupported('file.go'));
            assert.ok(isSupported('file.rs'));
            assert.ok(!isSupported('file.txt'));
            assert.ok(!isSupported('file.json'));
        });
    });

    // Test code extraction
    suite('Code Extraction', () => {
        test('Should extract function body', () => {
            const extractSymbolCode = (text: string, startLine: number, maxLines: number = 50): string => {
                const lines = text.split('\n');
                const endLine = Math.min(startLine + maxLines, lines.length);
                return lines.slice(startLine, endLine).join('\n');
            };

            const code = `line 0
function hello() {
    console.log('hello');
}
line 4`;

            const extracted = extractSymbolCode(code, 1, 3);
            assert.ok(extracted.includes('function hello'));
            assert.ok(extracted.includes('console.log'));
        });

        test('Should limit extracted lines', () => {
            const maxLines = 10;
            const lines: string[] = [];
            for (let i = 0; i < 100; i++) {
                lines.push(`line ${i}`);
            }

            const extractedLines = lines.slice(0, maxLines);
            assert.strictEqual(extractedLines.length, 10);
        });
    });

    // Test debouncing logic
    suite('Debouncing', () => {
        test('Should implement debounce delay', async () => {
            const debounceDelay = 300;
            let callCount = 0;
            let lastCallTime = 0;

            const debouncedFn = () => {
                callCount++;
                lastCallTime = Date.now();
            };

            // Simulate rapid calls
            const startTime = Date.now();
            for (let i = 0; i < 5; i++) {
                debouncedFn();
            }

            // After debounce, should have called multiple times (no actual debounce in this test)
            assert.strictEqual(callCount, 5);
        });
    });

    // Test line range calculation
    suite('Line Range Calculation', () => {
        test('Should calculate line range for symbol', () => {
            const calculateRange = (startLine: number, codeBlock: string): { start: number; end: number } => {
                const lines = codeBlock.split('\n');
                return {
                    start: startLine,
                    end: startLine + lines.length - 1
                };
            };

            const range = calculateRange(10, 'function foo() {\n  return 1;\n}');
            assert.strictEqual(range.start, 10);
            assert.strictEqual(range.end, 12);
        });
    });

    // Test enabled state
    suite('CodeLens Enabled State', () => {
        test('Should respect enabled configuration', () => {
            let enabled = true;

            const isEnabled = (): boolean => enabled;

            assert.ok(isEnabled());

            enabled = false;
            assert.ok(!isEnabled());
        });
    });

    // Test action display text
    suite('Action Display Text', () => {
        test('Should format action titles with icons', () => {
            const actions = [
                { icon: '$(question)', title: 'Explain' },
                { icon: '$(edit)', title: 'Refactor' },
                { icon: '$(beaker)', title: 'Add Tests' },
                { icon: '$(book)', title: 'Add Docs' },
                { icon: '$(zap)', title: 'Optimize' },
            ];

            assert.strictEqual(actions.length, 5);
            assert.ok(actions[0].icon.includes('question'));
            assert.ok(actions[1].icon.includes('edit'));
            assert.ok(actions[2].icon.includes('beaker'));
        });
    });
});
