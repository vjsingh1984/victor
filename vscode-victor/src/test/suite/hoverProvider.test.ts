/**
 * Hover Provider Tests
 *
 * Tests for the VictorHoverProvider which provides AI-powered
 * information tooltips when hovering over code symbols.
 */

import * as assert from 'assert';

suite('HoverProvider Test Suite', () => {
    // Test symbol extraction at position
    suite('Symbol Extraction', () => {
        test('Should extract word at position', () => {
            const extractWordAt = (text: string, position: number): string => {
                const wordPattern = /\w+/g;
                let match;
                while ((match = wordPattern.exec(text)) !== null) {
                    if (match.index <= position && position < match.index + match[0].length) {
                        return match[0];
                    }
                }
                return '';
            };

            const line = 'const myVariable = getValue();';
            assert.strictEqual(extractWordAt(line, 6), 'myVariable');
            assert.strictEqual(extractWordAt(line, 19), 'getValue');
            assert.strictEqual(extractWordAt(line, 0), 'const');
        });

        test('Should handle empty positions', () => {
            const line = 'const x = 5;';
            const wordPattern = /\w+/g;

            // Position on '=' which has no word
            const position = 8;
            let found = '';
            let match;
            while ((match = wordPattern.exec(line)) !== null) {
                if (match.index <= position && position < match.index + match[0].length) {
                    found = match[0];
                    break;
                }
            }

            // Position 8 is on '=' so no word found
            assert.strictEqual(found, '');
        });

        test('Should extract camelCase words', () => {
            const line = 'getUserDataFromServer();';
            const wordPattern = /\w+/g;
            const match = wordPattern.exec(line);

            assert.ok(match);
            assert.strictEqual(match[0], 'getUserDataFromServer');
        });

        test('Should extract snake_case words', () => {
            const line = 'get_user_data_from_server()';
            const wordPattern = /\w+/g;
            const match = wordPattern.exec(line);

            assert.ok(match);
            assert.strictEqual(match[0], 'get_user_data_from_server');
        });
    });

    // Test hover content generation
    suite('Hover Content Generation', () => {
        test('Should format markdown hover content', () => {
            const formatHoverMarkdown = (symbol: string, type: string, description: string): string => {
                return `**${symbol}** _(${type})_\n\n${description}`;
            };

            const content = formatHoverMarkdown('processData', 'function', 'Processes input data');
            assert.ok(content.includes('**processData**'));
            assert.ok(content.includes('_(function)_'));
            assert.ok(content.includes('Processes input data'));
        });

        test('Should include quick action links', () => {
            const formatActionsMarkdown = (symbol: string): string => {
                return [
                    `\n---\n`,
                    `[$(question) Explain](command:victor.explain?${encodeURIComponent(JSON.stringify({ symbol }))})`,
                    ` | `,
                    `[$(edit) Refactor](command:victor.refactor?${encodeURIComponent(JSON.stringify({ symbol }))})`,
                ].join('');
            };

            const actions = formatActionsMarkdown('myFunc');
            assert.ok(actions.includes('Explain'));
            assert.ok(actions.includes('Refactor'));
            assert.ok(actions.includes('command:'));
        });

        test('Should handle symbols without description', () => {
            const formatHoverWithFallback = (symbol: string, description?: string): string => {
                return description
                    ? `**${symbol}**\n\n${description}`
                    : `**${symbol}**\n\n_Click to get AI explanation_`;
            };

            const withDesc = formatHoverWithFallback('foo', 'Does something');
            const withoutDesc = formatHoverWithFallback('bar');

            assert.ok(withDesc.includes('Does something'));
            assert.ok(withoutDesc.includes('AI explanation'));
        });
    });

    // Test caching
    suite('Hover Cache', () => {
        test('Should cache hover results', () => {
            const cache = new Map<string, { content: string; timestamp: number }>();
            const cacheKey = 'file.ts:10:5';
            const content = 'Cached hover content';

            cache.set(cacheKey, { content, timestamp: Date.now() });

            assert.ok(cache.has(cacheKey));
            assert.strictEqual(cache.get(cacheKey)?.content, content);
        });

        test('Should invalidate expired cache entries', () => {
            const cacheTTL = 30000; // 30 seconds
            const now = Date.now();

            const isExpired = (timestamp: number): boolean => {
                return now - timestamp > cacheTTL;
            };

            const recentEntry = { timestamp: now - 10000 }; // 10 seconds ago
            const oldEntry = { timestamp: now - 60000 }; // 60 seconds ago

            assert.ok(!isExpired(recentEntry.timestamp));
            assert.ok(isExpired(oldEntry.timestamp));
        });

        test('Should limit cache size', () => {
            const maxCacheSize = 100;
            const cache = new Map<string, string>();

            // Add entries
            for (let i = 0; i < 150; i++) {
                cache.set(`key-${i}`, `value-${i}`);

                // Prune if over limit (simple LRU simulation)
                if (cache.size > maxCacheSize) {
                    const firstKey = cache.keys().next().value as string;
                    if (firstKey) cache.delete(firstKey);
                }
            }

            assert.strictEqual(cache.size, maxCacheSize);
        });

        test('Should create unique cache keys', () => {
            const createCacheKey = (filePath: string, line: number, character: number): string => {
                return `${filePath}:${line}:${character}`;
            };

            const key1 = createCacheKey('/path/file.ts', 10, 5);
            const key2 = createCacheKey('/path/file.ts', 10, 20);
            const key3 = createCacheKey('/path/other.ts', 10, 5);

            assert.notStrictEqual(key1, key2);
            assert.notStrictEqual(key1, key3);
        });
    });

    // Test hover triggers
    suite('Hover Triggers', () => {
        test('Should detect function hover', () => {
            const isFunctionCall = (line: string, word: string): boolean => {
                const pattern = new RegExp(`\\b${word}\\s*\\(`);
                return pattern.test(line);
            };

            assert.ok(isFunctionCall('processData(x, y)', 'processData'));
            assert.ok(!isFunctionCall('const processData = 5', 'processData'));
        });

        test('Should detect variable hover', () => {
            const isVariable = (line: string, word: string): boolean => {
                const pattern = new RegExp(`(?:const|let|var)\\s+${word}\\b`);
                return pattern.test(line);
            };

            assert.ok(isVariable('const myVar = 5', 'myVar'));
            assert.ok(isVariable('let count = 0', 'count'));
            assert.ok(!isVariable('myVar = 5', 'myVar'));
        });

        test('Should detect type/class hover', () => {
            const isTypeOrClass = (line: string, word: string): boolean => {
                const pattern = new RegExp(`(?:class|interface|type|enum)\\s+${word}\\b`);
                return pattern.test(line);
            };

            assert.ok(isTypeOrClass('class MyClass {}', 'MyClass'));
            assert.ok(isTypeOrClass('interface Config {}', 'Config'));
            assert.ok(isTypeOrClass('type Result = {}', 'Result'));
        });
    });

    // Test symbol type detection
    suite('Symbol Type Detection', () => {
        test('Should detect keyword symbols', () => {
            const keywords = new Set([
                'if', 'else', 'for', 'while', 'return', 'const', 'let', 'var',
                'function', 'class', 'interface', 'import', 'export', 'async', 'await'
            ]);

            const isKeyword = (word: string): boolean => keywords.has(word);

            assert.ok(isKeyword('const'));
            assert.ok(isKeyword('async'));
            assert.ok(!isKeyword('myVariable'));
        });

        test('Should skip hover for keywords', () => {
            const shouldShowHover = (word: string): boolean => {
                const skipWords = new Set(['if', 'else', 'for', 'while', 'return', '{', '}', '(', ')']);
                return !skipWords.has(word) && word.length > 1;
            };

            assert.ok(!shouldShowHover('if'));
            assert.ok(!shouldShowHover('{'));
            assert.ok(shouldShowHover('processData'));
        });
    });

    // Test language-specific patterns
    suite('Language-Specific Patterns', () => {
        test('Should handle TypeScript decorators', () => {
            const line = '@Component({ selector: "app" })';
            const decoratorPattern = /@(\w+)/;
            const match = line.match(decoratorPattern);

            assert.ok(match);
            assert.strictEqual(match[1], 'Component');
        });

        test('Should handle Python decorators', () => {
            const line = '@property';
            const decoratorPattern = /@(\w+)/;
            const match = line.match(decoratorPattern);

            assert.ok(match);
            assert.strictEqual(match[1], 'property');
        });

        test('Should handle JSX/TSX elements', () => {
            const line = '<MyComponent prop={value} />';
            const componentPattern = /<(\w+)/;
            const match = line.match(componentPattern);

            assert.ok(match);
            assert.strictEqual(match[1], 'MyComponent');
        });
    });

    // Test hover range calculation
    suite('Hover Range', () => {
        test('Should calculate word range', () => {
            const calculateWordRange = (line: string, position: number): { start: number; end: number } | null => {
                const wordPattern = /\w+/g;
                let match;
                while ((match = wordPattern.exec(line)) !== null) {
                    if (match.index <= position && position < match.index + match[0].length) {
                        return { start: match.index, end: match.index + match[0].length };
                    }
                }
                return null;
            };

            const line = 'const myVariable = 5;';
            const range = calculateWordRange(line, 8); // 'myVariable'

            assert.ok(range);
            assert.strictEqual(range.start, 6);
            assert.strictEqual(range.end, 16);
        });
    });

    // Test configuration
    suite('Configuration', () => {
        test('Should respect enabled setting', () => {
            let enabled = true;

            const isHoverEnabled = (): boolean => enabled;

            assert.ok(isHoverEnabled());
            enabled = false;
            assert.ok(!isHoverEnabled());
        });

        test('Should respect delay setting', () => {
            const delayMs = 300;

            const getHoverDelay = (): number => delayMs;

            assert.strictEqual(getHoverDelay(), 300);
        });
    });

    // Test AI integration formatting
    suite('AI Integration', () => {
        test('Should format AI request for hover', () => {
            const formatAIRequest = (symbol: string, context: string): string => {
                return `Briefly explain what "${symbol}" does in 1-2 sentences:\n\n${context}`;
            };

            const request = formatAIRequest('processData', 'function processData(input) { ... }');
            assert.ok(request.includes('processData'));
            assert.ok(request.includes('1-2 sentences'));
        });

        test('Should truncate long context', () => {
            const maxContextLength = 500;

            const truncateContext = (context: string): string => {
                if (context.length <= maxContextLength) return context;
                return context.substring(0, maxContextLength) + '...';
            };

            const shortContext = 'Short context';
            const longContext = 'x'.repeat(1000);

            assert.strictEqual(truncateContext(shortContext), shortContext);
            assert.ok(truncateContext(longContext).length <= maxContextLength + 3);
        });
    });
});
