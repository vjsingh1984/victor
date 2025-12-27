/**
 * Plans View Tests
 */

import * as assert from 'assert';
import {
    StepStatus,
    StepType,
    PlanStep,
    ExecutionPlan,
    createDemoPlan,
} from '../../plansView';

suite('PlansView Test Suite', () => {
    test('StepStatus enum values', () => {
        assert.strictEqual(StepStatus.Pending, 'pending');
        assert.strictEqual(StepStatus.InProgress, 'in_progress');
        assert.strictEqual(StepStatus.Completed, 'completed');
        assert.strictEqual(StepStatus.Skipped, 'skipped');
        assert.strictEqual(StepStatus.Failed, 'failed');
    });

    test('StepType enum values', () => {
        assert.strictEqual(StepType.Read, 'read');
        assert.strictEqual(StepType.Search, 'search');
        assert.strictEqual(StepType.Edit, 'edit');
        assert.strictEqual(StepType.Create, 'create');
        assert.strictEqual(StepType.Delete, 'delete');
        assert.strictEqual(StepType.Execute, 'execute');
        assert.strictEqual(StepType.Verify, 'verify');
    });

    test('PlanStep interface', () => {
        const step: PlanStep = {
            id: 'step-1',
            description: 'Read configuration file',
            type: StepType.Read,
            status: StepStatus.Pending,
            files: ['config.yaml'],
        };

        assert.strictEqual(step.id, 'step-1');
        assert.strictEqual(step.type, StepType.Read);
        assert.strictEqual(step.status, StepStatus.Pending);
        assert.ok(step.files?.includes('config.yaml'));
    });

    test('PlanStep with dependencies', () => {
        const step: PlanStep = {
            id: 'step-2',
            description: 'Apply changes based on config',
            type: StepType.Edit,
            status: StepStatus.Pending,
            dependencies: ['step-1'],
        };

        assert.ok(step.dependencies?.includes('step-1'));
    });

    test('PlanStep with output', () => {
        const step: PlanStep = {
            id: 'step-3',
            description: 'Search for usages',
            type: StepType.Search,
            status: StepStatus.Completed,
            output: 'Found 5 references',
        };

        assert.strictEqual(step.status, StepStatus.Completed);
        assert.strictEqual(step.output, 'Found 5 references');
    });

    test('PlanStep with error', () => {
        const step: PlanStep = {
            id: 'step-4',
            description: 'Run tests',
            type: StepType.Execute,
            status: StepStatus.Failed,
            error: 'Test assertion failed',
        };

        assert.strictEqual(step.status, StepStatus.Failed);
        assert.strictEqual(step.error, 'Test assertion failed');
    });
});

suite('ExecutionPlan Test Suite', () => {
    test('ExecutionPlan basic structure', () => {
        const plan: ExecutionPlan = {
            id: 'plan-123',
            goal: 'Implement user authentication',
            createdAt: Date.now(),
            status: 'draft',
            steps: [],
        };

        assert.strictEqual(plan.id, 'plan-123');
        assert.strictEqual(plan.status, 'draft');
        assert.strictEqual(plan.steps.length, 0);
    });

    test('ExecutionPlan status transitions', () => {
        const statuses: ExecutionPlan['status'][] = [
            'draft',
            'approved',
            'executing',
            'completed',
            'failed',
        ];

        statuses.forEach(status => {
            const plan: ExecutionPlan = {
                id: `plan-${status}`,
                goal: 'Test plan',
                createdAt: Date.now(),
                status,
                steps: [],
            };
            assert.strictEqual(plan.status, status);
        });
    });

    test('ExecutionPlan with approved timestamp', () => {
        const createdAt = Date.now() - 10000;
        const approvedAt = Date.now();

        const plan: ExecutionPlan = {
            id: 'plan-approved',
            goal: 'Approved plan',
            createdAt,
            approvedAt,
            status: 'approved',
            steps: [],
        };

        assert.ok(plan.approvedAt! > plan.createdAt);
    });

    test('ExecutionPlan with multiple steps', () => {
        const plan: ExecutionPlan = {
            id: 'plan-multi',
            goal: 'Multi-step plan',
            createdAt: Date.now(),
            status: 'executing',
            steps: [
                { id: 's1', description: 'Step 1', type: StepType.Read, status: StepStatus.Completed },
                { id: 's2', description: 'Step 2', type: StepType.Edit, status: StepStatus.InProgress },
                { id: 's3', description: 'Step 3', type: StepType.Verify, status: StepStatus.Pending },
            ],
        };

        assert.strictEqual(plan.steps.length, 3);

        const completed = plan.steps.filter(s => s.status === StepStatus.Completed);
        assert.strictEqual(completed.length, 1);

        const pending = plan.steps.filter(s => s.status === StepStatus.Pending);
        assert.strictEqual(pending.length, 1);
    });

    test('ExecutionPlan with metadata', () => {
        const plan: ExecutionPlan = {
            id: 'plan-meta',
            goal: 'Plan with metadata',
            createdAt: Date.now(),
            status: 'draft',
            steps: [],
            metadata: {
                author: 'victor',
                priority: 'high',
                tags: ['refactor', 'auth'],
            },
        };

        assert.strictEqual(plan.metadata?.author, 'victor');
        assert.strictEqual(plan.metadata?.priority, 'high');
    });
});

suite('createDemoPlan Test Suite', () => {
    test('Creates valid demo plan', () => {
        const demo = createDemoPlan();

        assert.ok(demo.id.startsWith('plan-'));
        assert.ok(demo.goal.length > 0);
        assert.strictEqual(demo.status, 'draft');
        assert.ok(demo.steps.length > 0);
    });

    test('Demo plan has various step types', () => {
        const demo = createDemoPlan();

        const types = new Set(demo.steps.map(s => s.type));

        assert.ok(types.has(StepType.Read));
        assert.ok(types.has(StepType.Search));
        assert.ok(types.has(StepType.Create));
        assert.ok(types.has(StepType.Edit));
        assert.ok(types.has(StepType.Verify));
    });

    test('Demo plan has various step statuses', () => {
        const demo = createDemoPlan();

        const statuses = new Set(demo.steps.map(s => s.status));

        assert.ok(statuses.has(StepStatus.Completed));
        assert.ok(statuses.has(StepStatus.InProgress));
        assert.ok(statuses.has(StepStatus.Pending));
    });

    test('Demo plan steps have dependencies', () => {
        const demo = createDemoPlan();

        const stepsWithDeps = demo.steps.filter(s => s.dependencies && s.dependencies.length > 0);
        assert.ok(stepsWithDeps.length > 0);
    });

    test('Demo plan steps have files', () => {
        const demo = createDemoPlan();

        const stepsWithFiles = demo.steps.filter(s => s.files && s.files.length > 0);
        assert.ok(stepsWithFiles.length > 0);
    });
});

suite('Plan Step Progress Calculation', () => {
    test('Calculate progress for empty plan', () => {
        const plan: ExecutionPlan = {
            id: 'progress-0',
            goal: 'Empty plan',
            createdAt: Date.now(),
            status: 'draft',
            steps: [],
        };

        const progress = plan.steps.length === 0 ? 0 :
            plan.steps.filter(s => s.status === StepStatus.Completed).length / plan.steps.length * 100;

        assert.strictEqual(progress, 0);
    });

    test('Calculate progress for partially complete plan', () => {
        const plan: ExecutionPlan = {
            id: 'progress-50',
            goal: 'Half done plan',
            createdAt: Date.now(),
            status: 'executing',
            steps: [
                { id: 's1', description: 'Done', type: StepType.Read, status: StepStatus.Completed },
                { id: 's2', description: 'Pending', type: StepType.Edit, status: StepStatus.Pending },
            ],
        };

        const progress = plan.steps.filter(s => s.status === StepStatus.Completed).length / plan.steps.length * 100;
        assert.strictEqual(progress, 50);
    });

    test('Calculate progress for complete plan', () => {
        const plan: ExecutionPlan = {
            id: 'progress-100',
            goal: 'Complete plan',
            createdAt: Date.now(),
            status: 'completed',
            steps: [
                { id: 's1', description: 'Done 1', type: StepType.Read, status: StepStatus.Completed },
                { id: 's2', description: 'Done 2', type: StepType.Edit, status: StepStatus.Completed },
            ],
        };

        const progress = plan.steps.filter(s => s.status === StepStatus.Completed).length / plan.steps.length * 100;
        assert.strictEqual(progress, 100);
    });
});
