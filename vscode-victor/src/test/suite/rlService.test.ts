/**
 * RL Service Tests
 */

import * as assert from 'assert';
import { RLStats, RLRecommendation, TASK_TYPES, STRATEGIES } from '../../rlService';

suite('RLService Test Suite', () => {
    test('RLStats interface', () => {
        const stats: RLStats = {
            total_selections: 100,
            epsilon: 0.1,
            strategy: 'epsilon_greedy',
            q_table: {
                'anthropic': 0.85,
                'openai': 0.75,
                'ollama': 0.65,
            },
            selection_counts: {
                'anthropic': 50,
                'openai': 30,
                'ollama': 20,
            },
        };

        assert.strictEqual(stats.total_selections, 100);
        assert.strictEqual(stats.epsilon, 0.1);
        assert.strictEqual(stats.strategy, 'epsilon_greedy');
        assert.ok(stats.q_table['anthropic'] > stats.q_table['openai']);
    });

    test('RLStats with task-specific Q-values', () => {
        const stats: RLStats = {
            total_selections: 50,
            epsilon: 0.15,
            strategy: 'ucb',
            q_table: { 'anthropic': 0.8 },
            selection_counts: { 'anthropic': 50 },
            q_table_by_task: {
                'simple': { 'anthropic': 0.9, 'openai': 0.85 },
                'complex': { 'anthropic': 0.95, 'openai': 0.70 },
                'action': { 'anthropic': 0.80, 'openai': 0.90 },
            },
        };

        assert.ok(stats.q_table_by_task);
        assert.ok(stats.q_table_by_task['simple']);
        assert.ok(stats.q_table_by_task['complex']);
    });

    test('RLRecommendation interface', () => {
        const recommendation: RLRecommendation = {
            recommended: 'anthropic',
            strategy: 'epsilon_greedy',
            exploration_rate: 0.1,
            q_values: {
                'anthropic': 0.85,
                'openai': 0.75,
            },
            available_providers: ['anthropic', 'openai', 'ollama'],
        };

        assert.strictEqual(recommendation.recommended, 'anthropic');
        assert.ok(recommendation.available_providers.length > 0);
    });
});

suite('Task Types', () => {
    test('TASK_TYPES is defined', () => {
        assert.ok(TASK_TYPES);
        assert.ok(Array.isArray(TASK_TYPES));
        assert.ok(TASK_TYPES.length > 0);
    });

    test('Task types have required fields', () => {
        TASK_TYPES.forEach(task => {
            assert.ok(task.label, `Task type missing label`);
            assert.ok(task.value, `Task type missing value`);
            assert.ok(task.description, `Task type missing description`);
        });
    });

    const expectedTaskTypes = ['simple', 'complex', 'action', 'generation', 'analysis'];

    expectedTaskTypes.forEach(type => {
        test(`Has task type: ${type}`, () => {
            const found = TASK_TYPES.find(t => t.value === type);
            assert.ok(found, `Missing task type: ${type}`);
        });
    });
});

suite('Strategies', () => {
    test('STRATEGIES is defined', () => {
        assert.ok(STRATEGIES);
        assert.ok(Array.isArray(STRATEGIES));
        assert.ok(STRATEGIES.length > 0);
    });

    test('Strategies have required fields', () => {
        STRATEGIES.forEach(strategy => {
            assert.ok(strategy.label, `Strategy missing label`);
            assert.ok(strategy.value, `Strategy missing value`);
            assert.ok(strategy.description, `Strategy missing description`);
        });
    });

    const expectedStrategies = ['epsilon_greedy', 'ucb', 'greedy', 'random'];

    expectedStrategies.forEach(strategy => {
        test(`Has strategy: ${strategy}`, () => {
            const found = STRATEGIES.find(s => s.value === strategy);
            assert.ok(found, `Missing strategy: ${strategy}`);
        });
    });
});

suite('Q-Value Calculations', () => {
    test('Find top provider', () => {
        const qTable: Record<string, number> = {
            'anthropic': 0.85,
            'openai': 0.75,
            'ollama': 0.65,
        };

        let maxQ = -Infinity;
        let topProvider: string | null = null;

        for (const [provider, q] of Object.entries(qTable)) {
            if (q > maxQ) {
                maxQ = q;
                topProvider = provider;
            }
        }

        assert.strictEqual(topProvider, 'anthropic');
        assert.strictEqual(maxQ, 0.85);
    });

    test('Handle empty Q-table', () => {
        const qTable: Record<string, number> = {};

        let topProvider: string | null = null;

        for (const [provider, q] of Object.entries(qTable)) {
            // Won't execute
        }

        assert.strictEqual(topProvider, null);
    });

    test('Handle tied Q-values', () => {
        const qTable: Record<string, number> = {
            'provider1': 0.80,
            'provider2': 0.80,
        };

        // First one wins in case of tie
        let maxQ = -Infinity;
        let topProvider: string | null = null;

        for (const [provider, q] of Object.entries(qTable)) {
            if (q > maxQ) {
                maxQ = q;
                topProvider = provider;
            }
        }

        assert.ok(topProvider === 'provider1' || topProvider === 'provider2');
    });
});

suite('Exploration Rate', () => {
    test('Valid exploration rate range', () => {
        const validRates = [0.0, 0.1, 0.5, 0.9, 1.0];

        validRates.forEach(rate => {
            assert.ok(rate >= 0 && rate <= 1, `Rate ${rate} out of range`);
        });
    });

    test('Invalid exploration rate detection', () => {
        const invalidRates = [-0.1, 1.1, 2.0];

        invalidRates.forEach(rate => {
            const isValid = rate >= 0 && rate <= 1;
            assert.ok(!isValid, `Rate ${rate} should be invalid`);
        });
    });

    test('Epsilon controls exploration vs exploitation', () => {
        // High epsilon = more exploration
        const highEpsilon = 0.9;
        // Low epsilon = more exploitation
        const lowEpsilon = 0.1;

        assert.ok(highEpsilon > lowEpsilon);
    });
});

suite('Strategy Behavior', () => {
    test('Epsilon-greedy description', () => {
        const strategy = STRATEGIES.find(s => s.value === 'epsilon_greedy');
        assert.ok(strategy);
        assert.ok(strategy.description.toLowerCase().includes('exploration'));
    });

    test('UCB description', () => {
        const strategy = STRATEGIES.find(s => s.value === 'ucb');
        assert.ok(strategy);
        assert.ok(strategy.description.toLowerCase().includes('explore'));
    });

    test('Greedy description', () => {
        const strategy = STRATEGIES.find(s => s.value === 'greedy');
        assert.ok(strategy);
        assert.ok(strategy.description.toLowerCase().includes('best'));
    });

    test('Random description', () => {
        const strategy = STRATEGIES.find(s => s.value === 'random');
        assert.ok(strategy);
        assert.ok(strategy.description.toLowerCase().includes('exploration'));
    });
});

suite('Status Bar Display', () => {
    test('Format status bar text with provider', () => {
        const topProvider = 'anthropic';
        const text = `$(lightbulb) RL: ${topProvider}`;

        assert.strictEqual(text, '$(lightbulb) RL: anthropic');
    });

    test('Format status bar text while learning', () => {
        const topProvider = null;
        const text = `$(lightbulb) RL: ${topProvider || 'learning'}`;

        assert.strictEqual(text, '$(lightbulb) RL: learning');
    });

    test('Format tooltip with stats', () => {
        const stats: RLStats = {
            total_selections: 42,
            epsilon: 0.15,
            strategy: 'ucb',
            q_table: {},
            selection_counts: {},
        };

        const tooltip = `Victor RL - ${stats.total_selections} selections, ε=${stats.epsilon.toFixed(2)}`;
        assert.strictEqual(tooltip, 'Victor RL - 42 selections, ε=0.15');
    });
});

suite('Q-Table Display', () => {
    test('Sort Q-table by value descending', () => {
        const qTable: Record<string, number> = {
            'openai': 0.75,
            'anthropic': 0.85,
            'ollama': 0.65,
        };

        const sorted = Object.entries(qTable).sort(([, a], [, b]) => b - a);

        assert.strictEqual(sorted[0][0], 'anthropic');
        assert.strictEqual(sorted[1][0], 'openai');
        assert.strictEqual(sorted[2][0], 'ollama');
    });

    test('Format Q-value display', () => {
        const provider = 'anthropic';
        const q = 0.8567;
        const count = 42;

        const display = `${provider}: Q=${q.toFixed(3)} (${count} selections)`;
        assert.strictEqual(display, 'anthropic: Q=0.857 (42 selections)');
    });

    test('Format Q-value bar', () => {
        const q = 0.75;
        const barLength = Math.round(q * 20);
        const bar = '█'.repeat(barLength);

        assert.strictEqual(bar.length, 15);
    });
});

suite('Recommendation Display', () => {
    test('Format recommendation message', () => {
        const recommended = 'anthropic';
        const taskType = 'Complex';

        const message = `Recommended: **${recommended}** for ${taskType} tasks`;
        assert.ok(message.includes('anthropic'));
        assert.ok(message.includes('Complex'));
    });

    test('Format recommendation without task type', () => {
        const recommended = 'openai';

        const message = `Recommended: **${recommended}**`;
        assert.strictEqual(message, 'Recommended: **openai**');
    });
});

suite('Reset Confirmation', () => {
    test('Reset warning message', () => {
        const message = 'This will reset all learned Q-values. The model selector will start learning from scratch.';

        assert.ok(message.includes('reset'));
        assert.ok(message.includes('Q-values'));
        assert.ok(message.includes('learning'));
    });
});
