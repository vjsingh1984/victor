/**
 * Agents View Tests
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import {
    AgentStatus,
    AgentTask,
    AgentToolCall,
    AgentsViewProvider,
    createDemoAgent,
} from '../../agentsView';

// Mock VictorClient
class MockVictorClient {
    private _agents: any[] = [];
    private _eventHandlers: ((event: any) => void)[] = [];

    async listAgents(): Promise<any[]> {
        return this._agents;
    }

    async startAgent(task: string, mode: string): Promise<string> {
        const id = `agent-${Date.now()}`;
        this._agents.push({
            id,
            task,
            mode,
            status: AgentStatus.Running,
            start_time: Date.now() / 1000,
            tool_calls: [],
        });
        return id;
    }

    async cancelAgent(id: string): Promise<void> {
        const agent = this._agents.find(a => a.id === id);
        if (agent) {
            agent.status = AgentStatus.Cancelled;
        }
    }

    async clearAgents(): Promise<void> {
        this._agents = this._agents.filter(
            a => a.status === AgentStatus.Running || a.status === AgentStatus.Pending
        );
    }

    onAgentEvent(handler: (event: any) => void): void {
        this._eventHandlers.push(handler);
    }

    // Test helper to simulate events
    simulateEvent(event: any): void {
        this._eventHandlers.forEach(h => h(event));
    }

    setAgents(agents: any[]): void {
        this._agents = agents;
    }
}

suite('AgentsView Test Suite', () => {
    let mockClient: MockVictorClient;
    let agentsView: AgentsViewProvider;

    setup(() => {
        mockClient = new MockVictorClient();
        agentsView = new AgentsViewProvider(mockClient as any);
    });

    teardown(() => {
        agentsView.dispose();
    });

    test('Should create AgentsViewProvider', () => {
        assert.ok(agentsView);
    });

    test('Should return empty state when no agents', async () => {
        const children = await agentsView.getChildren();
        assert.ok(children.length === 1); // Info item
    });

    test('Should add and track agent', () => {
        const task: AgentTask = {
            id: 'test-agent-1',
            name: 'Test Agent',
            description: 'Testing agent functionality',
            status: AgentStatus.Running,
            startTime: Date.now(),
            toolCalls: [],
            mode: 'build',
        };

        agentsView.addAgent(task);
        assert.strictEqual(agentsView.getActiveCount(), 1);
    });

    test('Should update agent status', () => {
        const task: AgentTask = {
            id: 'test-agent-2',
            name: 'Test Agent 2',
            description: 'Testing',
            status: AgentStatus.Running,
            startTime: Date.now(),
            toolCalls: [],
        };

        agentsView.addAgent(task);
        agentsView.updateAgent('test-agent-2', { status: AgentStatus.Completed });

        // After completing, active count should be 0
        assert.strictEqual(agentsView.getActiveCount(), 0);
    });

    test('Should add tool call to agent', () => {
        const task: AgentTask = {
            id: 'test-agent-3',
            name: 'Test Agent 3',
            description: 'Testing',
            status: AgentStatus.Running,
            startTime: Date.now(),
            toolCalls: [],
        };

        agentsView.addAgent(task);

        const toolCall: AgentToolCall = {
            id: 'tc-1',
            name: 'read_file',
            status: 'running',
            startTime: Date.now(),
        };

        agentsView.addToolCall('test-agent-3', toolCall);
        // Verify by checking tree structure
    });

    test('Should update tool call status', () => {
        const task: AgentTask = {
            id: 'test-agent-4',
            name: 'Test Agent 4',
            description: 'Testing',
            status: AgentStatus.Running,
            startTime: Date.now(),
            toolCalls: [{
                id: 'tc-1',
                name: 'read_file',
                status: 'running',
                startTime: Date.now(),
            }],
        };

        agentsView.addAgent(task);
        agentsView.updateToolCall('test-agent-4', 'tc-1', {
            status: 'success',
            endTime: Date.now(),
            result: 'File read successfully',
        });
    });

    test('Should remove agent', () => {
        const task: AgentTask = {
            id: 'test-agent-5',
            name: 'Test Agent 5',
            description: 'Testing',
            status: AgentStatus.Running,
            startTime: Date.now(),
            toolCalls: [],
        };

        agentsView.addAgent(task);
        assert.strictEqual(agentsView.getActiveCount(), 1);

        agentsView.removeAgent('test-agent-5');
        assert.strictEqual(agentsView.getActiveCount(), 0);
    });

    test('Should count active agents correctly', () => {
        const runningTask: AgentTask = {
            id: 'running-1',
            name: 'Running',
            description: '',
            status: AgentStatus.Running,
            startTime: Date.now(),
            toolCalls: [],
        };

        const pendingTask: AgentTask = {
            id: 'pending-1',
            name: 'Pending',
            description: '',
            status: AgentStatus.Pending,
            startTime: Date.now(),
            toolCalls: [],
        };

        const completedTask: AgentTask = {
            id: 'completed-1',
            name: 'Completed',
            description: '',
            status: AgentStatus.Completed,
            startTime: Date.now(),
            toolCalls: [],
        };

        agentsView.addAgent(runningTask);
        agentsView.addAgent(pendingTask);
        agentsView.addAgent(completedTask);

        // Only running and pending count as active
        assert.strictEqual(agentsView.getActiveCount(), 2);
    });

    test('Should refresh tree data', () => {
        let refreshed = false;
        agentsView.onDidChangeTreeData(() => {
            refreshed = true;
        });

        agentsView.refresh();
        assert.ok(refreshed);
    });

    test('createDemoAgent should return valid agent', () => {
        const demo = createDemoAgent();

        assert.ok(demo.id);
        assert.ok(demo.name);
        assert.strictEqual(demo.status, AgentStatus.Running);
        assert.ok(demo.toolCalls.length > 0);
        assert.strictEqual(demo.mode, 'explore');
    });
});

suite('AgentStatus Test Suite', () => {
    test('Should have all expected status values', () => {
        assert.strictEqual(AgentStatus.Pending, 'pending');
        assert.strictEqual(AgentStatus.Running, 'running');
        assert.strictEqual(AgentStatus.Paused, 'paused');
        assert.strictEqual(AgentStatus.Completed, 'completed');
        assert.strictEqual(AgentStatus.Error, 'error');
        assert.strictEqual(AgentStatus.Cancelled, 'cancelled');
    });
});

suite('AgentTask Interface Test Suite', () => {
    test('Should create valid AgentTask', () => {
        const task: AgentTask = {
            id: 'task-1',
            name: 'Test Task',
            description: 'A test task',
            status: AgentStatus.Running,
            progress: 50,
            startTime: Date.now(),
            endTime: undefined,
            toolCalls: [],
            output: 'Some output',
            error: undefined,
            mode: 'build',
        };

        assert.strictEqual(task.id, 'task-1');
        assert.strictEqual(task.progress, 50);
        assert.strictEqual(task.mode, 'build');
    });

    test('Should create AgentToolCall', () => {
        const toolCall: AgentToolCall = {
            id: 'tc-1',
            name: 'read_file',
            status: 'success',
            startTime: Date.now() - 1000,
            endTime: Date.now(),
            result: 'File contents',
        };

        assert.strictEqual(toolCall.name, 'read_file');
        assert.strictEqual(toolCall.status, 'success');
        assert.ok(toolCall.endTime! > toolCall.startTime);
    });
});
