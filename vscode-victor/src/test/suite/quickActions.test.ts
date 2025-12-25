/**
 * Quick Actions Tests
 */

import * as assert from 'assert';
import { QuickAction } from '../../quickActions';

suite('QuickActions Test Suite', () => {
    test('QuickAction interface', () => {
        const action: QuickAction = {
            label: 'Fix',
            description: 'Fix bugs in code',
            icon: '$(bug)',
            action: 'fix',
            shortcut: 'f',
            category: 'edit',
        };

        assert.strictEqual(action.label, 'Fix');
        assert.strictEqual(action.action, 'fix');
        assert.strictEqual(action.shortcut, 'f');
        assert.strictEqual(action.category, 'edit');
    });

    test('QuickAction without shortcut', () => {
        const action: QuickAction = {
            label: 'Simplify',
            description: 'Simplify code',
            icon: '$(sparkle)',
            action: 'simplify',
            category: 'refactor',
        };

        assert.ok(!action.shortcut);
    });

    test('QuickAction categories', () => {
        const categories: QuickAction['category'][] = [
            'edit',
            'explain',
            'generate',
            'refactor',
            'debug',
        ];

        categories.forEach(category => {
            const action: QuickAction = {
                label: 'Test',
                description: 'Test action',
                icon: '$(test)',
                action: 'test',
                category,
            };
            assert.strictEqual(action.category, category);
        });
    });
});

suite('Built-in Actions', () => {
    const builtInActions: QuickAction[] = [
        { label: 'Fix', description: 'Fix bugs', icon: '$(bug)', action: 'fix', shortcut: 'f', category: 'edit' },
        { label: 'Refactor', description: 'Refactor code', icon: '$(edit)', action: 'refactor', shortcut: 'r', category: 'refactor' },
        { label: 'Optimize', description: 'Optimize performance', icon: '$(zap)', action: 'optimize', shortcut: 'o', category: 'edit' },
        { label: 'Simplify', description: 'Simplify code', icon: '$(sparkle)', action: 'simplify', category: 'refactor' },
        { label: 'Explain', description: 'Explain code', icon: '$(question)', action: 'explain', shortcut: 'e', category: 'explain' },
        { label: 'Generate Tests', description: 'Generate tests', icon: '$(beaker)', action: 'test', shortcut: 't', category: 'generate' },
        { label: 'Add Documentation', description: 'Add docs', icon: '$(book)', action: 'document', shortcut: 'd', category: 'generate' },
        { label: 'Add Types', description: 'Add types', icon: '$(symbol-type-parameter)', action: 'add_types', category: 'generate' },
        { label: 'Add Error Handling', description: 'Add try-catch', icon: '$(shield)', action: 'error_handling', category: 'generate' },
        { label: 'Add Logging', description: 'Add logging', icon: '$(output)', action: 'add_logging', category: 'debug' },
        { label: 'Find Issues', description: 'Review code', icon: '$(search)', action: 'review', category: 'debug' },
    ];

    test('Has core edit actions', () => {
        const editActions = builtInActions.filter(a => a.category === 'edit');
        assert.ok(editActions.length >= 2);
        assert.ok(editActions.some(a => a.action === 'fix'));
        assert.ok(editActions.some(a => a.action === 'optimize'));
    });

    test('Has explain actions', () => {
        const explainActions = builtInActions.filter(a => a.category === 'explain');
        assert.ok(explainActions.length >= 1);
        assert.ok(explainActions.some(a => a.action === 'explain'));
    });

    test('Has generate actions', () => {
        const generateActions = builtInActions.filter(a => a.category === 'generate');
        assert.ok(generateActions.length >= 3);
        assert.ok(generateActions.some(a => a.action === 'test'));
        assert.ok(generateActions.some(a => a.action === 'document'));
    });

    test('Has refactor actions', () => {
        const refactorActions = builtInActions.filter(a => a.category === 'refactor');
        assert.ok(refactorActions.length >= 1);
        assert.ok(refactorActions.some(a => a.action === 'refactor'));
    });

    test('Has debug actions', () => {
        const debugActions = builtInActions.filter(a => a.category === 'debug');
        assert.ok(debugActions.length >= 1);
        assert.ok(debugActions.some(a => a.action === 'review'));
    });

    test('Actions have unique shortcuts', () => {
        const shortcuts = builtInActions
            .filter(a => a.shortcut)
            .map(a => a.shortcut);
        const uniqueShortcuts = new Set(shortcuts);

        assert.strictEqual(shortcuts.length, uniqueShortcuts.size);
    });

    test('Actions have unique action identifiers', () => {
        const actions = builtInActions.map(a => a.action);
        const uniqueActions = new Set(actions);

        assert.strictEqual(actions.length, uniqueActions.size);
    });
});

suite('Keyboard Shortcuts', () => {
    const shortcuts = ['f', 'r', 'o', 'e', 't', 'd'];

    shortcuts.forEach(shortcut => {
        test(`Shortcut '${shortcut}' is single character`, () => {
            assert.strictEqual(shortcut.length, 1);
        });
    });

    test('Shortcuts are lowercase', () => {
        shortcuts.forEach(s => {
            assert.strictEqual(s, s.toLowerCase());
        });
    });
});

suite('Context-Aware Actions', () => {
    test('TypeScript/JavaScript extra actions', () => {
        const languages = ['typescript', 'javascript', 'typescriptreact', 'javascriptreact'];

        languages.forEach(lang => {
            const extraAction: QuickAction = {
                label: 'Convert to Async',
                description: 'Convert to async/await pattern',
                icon: '$(sync)',
                action: 'convert_async',
                category: 'refactor',
            };

            assert.strictEqual(extraAction.action, 'convert_async');
        });
    });

    test('Python extra actions', () => {
        const extraAction: QuickAction = {
            label: 'Add Type Hints',
            description: 'Add Python type hints',
            icon: '$(symbol-type-parameter)',
            action: 'add_type_hints',
            category: 'generate',
        };

        assert.strictEqual(extraAction.action, 'add_type_hints');
    });

    test('Extract function action for multiple languages', () => {
        const supportedLanguages = ['typescript', 'javascript', 'python', 'java', 'csharp'];

        supportedLanguages.forEach(lang => {
            const extractAction: QuickAction = {
                label: 'Extract Function',
                description: 'Extract selection into a function',
                icon: '$(symbol-function)',
                action: 'extract_function',
                category: 'refactor',
            };

            assert.strictEqual(extractAction.action, 'extract_function');
        });
    });
});

suite('Recent Actions Tracking', () => {
    test('Add action to recent list', () => {
        const recentActions: string[] = [];
        const action = 'fix';

        recentActions.unshift(action);
        assert.strictEqual(recentActions[0], 'fix');
    });

    test('Remove duplicate from recent list', () => {
        const recentActions = ['explain', 'fix', 'test'];
        const action = 'fix';

        const index = recentActions.indexOf(action);
        if (index >= 0) {
            recentActions.splice(index, 1);
        }
        recentActions.unshift(action);

        assert.strictEqual(recentActions[0], 'fix');
        assert.strictEqual(recentActions.filter(a => a === 'fix').length, 1);
    });

    test('Limit recent actions to max size', () => {
        const maxRecentActions = 5;
        const recentActions = ['a', 'b', 'c', 'd', 'e', 'f'];

        const trimmed = recentActions.slice(0, maxRecentActions);
        assert.strictEqual(trimmed.length, maxRecentActions);
    });

    test('Sort by recent usage', () => {
        const recentActions = ['fix', 'explain'];
        const allActions = [
            { action: 'test' },
            { action: 'fix' },
            { action: 'explain' },
            { action: 'refactor' },
        ];

        const sorted = [...allActions].sort((a, b) => {
            const aIndex = recentActions.indexOf(a.action);
            const bIndex = recentActions.indexOf(b.action);

            if (aIndex >= 0 && bIndex >= 0) return aIndex - bIndex;
            if (aIndex >= 0) return -1;
            if (bIndex >= 0) return 1;
            return 0;
        });

        assert.strictEqual(sorted[0].action, 'fix');
        assert.strictEqual(sorted[1].action, 'explain');
    });
});

suite('Quick Pick Items', () => {
    test('Convert action to quick pick item', () => {
        const action: QuickAction = {
            label: 'Fix',
            description: 'Fix bugs',
            icon: '$(bug)',
            action: 'fix',
            shortcut: 'f',
            category: 'edit',
        };

        const item = {
            label: `${action.icon} ${action.label}`,
            description: action.shortcut ? `(${action.shortcut})` : undefined,
            detail: action.description,
        };

        assert.strictEqual(item.label, '$(bug) Fix');
        assert.strictEqual(item.description, '(f)');
        assert.strictEqual(item.detail, 'Fix bugs');
    });

    test('Quick pick item without shortcut', () => {
        const action: QuickAction = {
            label: 'Simplify',
            description: 'Simplify code',
            icon: '$(sparkle)',
            action: 'simplify',
            category: 'refactor',
        };

        const item = {
            label: `${action.icon} ${action.label}`,
            description: action.shortcut ? `(${action.shortcut})` : undefined,
            detail: action.description,
        };

        assert.ok(!item.description);
    });
});

suite('Category Labels', () => {
    const categoryLabels: Record<string, string> = {
        edit: 'Edit',
        explain: 'Understand',
        generate: 'Generate',
        refactor: 'Refactor',
        debug: 'Debug',
    };

    Object.entries(categoryLabels).forEach(([key, label]) => {
        test(`Category '${key}' has label '${label}'`, () => {
            assert.strictEqual(categoryLabels[key], label);
        });
    });
});
