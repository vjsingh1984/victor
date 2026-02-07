/**
 * Agents View Provider
 *
 * Provides a TreeView for managing and monitoring AI agents.
 * Displays active agents, their status, and allows control.
 *
 * Similar to Cursor's agent sidebar, this shows:
 * - Active agent tasks
 * - Agent progress and status
 * - Tool calls made by agents
 * - Agent output/results
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

// Agent status enum
export enum AgentStatus {
    Pending = 'pending',
    Running = 'running',
    Paused = 'paused',
    Completed = 'completed',
    Error = 'error',
    Cancelled = 'cancelled',
}

// Agent task representation
export interface AgentTask {
    id: string;
    name: string;
    description: string;
    status: AgentStatus;
    progress?: number;  // 0-100
    startTime: number;
    endTime?: number;
    toolCalls: AgentToolCall[];
    output?: string;
    error?: string;
    mode?: 'build' | 'plan' | 'explore';
}

// Tool call made by an agent
export interface AgentToolCall {
    id: string;
    name: string;
    status: 'pending' | 'running' | 'success' | 'error';
    startTime: number;
    endTime?: number;
    result?: string;
}

// WebSocket event data interfaces
interface AgentStartedEvent {
    type: 'agent_started';
    data: AgentTask;
}

interface AgentRunningEvent {
    type: 'agent_running';
    data: Partial<AgentTask>;
}

interface AgentToolCallEvent {
    type: 'agent_tool_call';
    data: {
        agent_id: string;
        tool_call: {
            id: string;
            name: string;
            arguments: string;
            status: 'pending' | 'running' | 'success' | 'error';
            startTime?: number;
            endTime?: number;
            result?: string;
        };
    };
}

interface AgentToolResultEvent {
    type: 'agent_tool_result';
    data: {
        agent_id: string;
        tool_call_id: string;
        result: string;
        error?: string;
        status?: 'success' | 'error';
    };
}

interface AgentCompletedEvent {
    type: 'agent_completed' | 'agent_error' | 'agent_cancelled';
    data: Partial<AgentTask>;
}

type AgentEventData = AgentStartedEvent | AgentRunningEvent | AgentToolCallEvent | AgentToolResultEvent | AgentCompletedEvent;

// Backend agent data format (from Victor server)
interface BackendAgentData {
    id: string;
    name?: string;
    task?: string;
    description?: string;
    status: string;
    progress?: number;
    start_time: number;
    end_time?: number;
    mode?: 'build' | 'plan' | 'explore';
    output?: string;
    error?: string;
    tool_calls?: BackendToolCallData[];
    agent_id?: string;
}

interface BackendToolCallData {
    id: string;
    name: string;
    status: string;
    start_time: number;
    end_time?: number;
    result?: string;
}

// Tree item types
type AgentTreeItem = AgentTaskItem | AgentToolCallItem | AgentInfoItem;

/**
 * Tree item representing an agent task
 */
class AgentTaskItem extends vscode.TreeItem {
    constructor(
        public readonly task: AgentTask,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(task.name, collapsibleState);

        this.id = task.id;
        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = `agent-${task.status}`;
    }

    private _getDescription(): string {
        const elapsed = this._getElapsedTime();
        const progress = this.task.progress !== undefined ? ` (${this.task.progress}%)` : '';
        return `${this._getStatusLabel()}${progress} • ${elapsed}`;
    }

    private _getStatusLabel(): string {
        switch (this.task.status) {
            case AgentStatus.Pending: return '⏳ Pending';
            case AgentStatus.Running: return '🔄 Running';
            case AgentStatus.Paused: return '⏸️ Paused';
            case AgentStatus.Completed: return '✅ Done';
            case AgentStatus.Error: return '❌ Error';
            case AgentStatus.Cancelled: return '🚫 Cancelled';
            default: return this.task.status;
        }
    }

    private _getElapsedTime(): string {
        const end = this.task.endTime || Date.now();
        const elapsed = end - this.task.startTime;

        if (elapsed < 1000) {return '<1s';}
        if (elapsed < 60000) {return `${Math.floor(elapsed / 1000)}s`;}
        if (elapsed < 3600000) {return `${Math.floor(elapsed / 60000)}m`;}
        return `${Math.floor(elapsed / 3600000)}h`;
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### ${this.task.name}\n\n`);
        md.appendMarkdown(`**Status:** ${this._getStatusLabel()}\n\n`);
        md.appendMarkdown(`**Mode:** ${this.task.mode || 'build'}\n\n`);
        md.appendMarkdown(`**Description:** ${this.task.description}\n\n`);
        md.appendMarkdown(`**Tool Calls:** ${this.task.toolCalls.length}\n\n`);

        if (this.task.error) {
            md.appendMarkdown(`**Error:** ${this.task.error}\n\n`);
        }

        return md;
    }

    private _getIcon(): vscode.ThemeIcon {
        switch (this.task.status) {
            case AgentStatus.Pending:
                return new vscode.ThemeIcon('clock', new vscode.ThemeColor('charts.yellow'));
            case AgentStatus.Running:
                return new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('charts.blue'));
            case AgentStatus.Paused:
                return new vscode.ThemeIcon('debug-pause', new vscode.ThemeColor('charts.orange'));
            case AgentStatus.Completed:
                return new vscode.ThemeIcon('check', new vscode.ThemeColor('charts.green'));
            case AgentStatus.Error:
                return new vscode.ThemeIcon('error', new vscode.ThemeColor('charts.red'));
            case AgentStatus.Cancelled:
                return new vscode.ThemeIcon('circle-slash', new vscode.ThemeColor('charts.gray'));
            default:
                return new vscode.ThemeIcon('robot');
        }
    }
}

/**
 * Tree item representing a tool call within an agent
 */
class AgentToolCallItem extends vscode.TreeItem {
    constructor(
        public readonly toolCall: AgentToolCall,
        public readonly parentId: string
    ) {
        super(toolCall.name, vscode.TreeItemCollapsibleState.None);

        this.id = `${parentId}-${toolCall.id}`;
        this.description = this._getDescription();
        this.iconPath = this._getIcon();
        this.contextValue = 'agent-tool-call';

        if (toolCall.result) {
            this.tooltip = new vscode.MarkdownString(`\`\`\`\n${toolCall.result.slice(0, 500)}\n\`\`\``);
        }
    }

    private _getDescription(): string {
        const elapsed = this._getElapsedTime();
        return `${this._getStatusIcon()} ${elapsed}`;
    }

    private _getStatusIcon(): string {
        switch (this.toolCall.status) {
            case 'pending': return '⏳';
            case 'running': return '🔄';
            case 'success': return '✅';
            case 'error': return '❌';
            default: return '•';
        }
    }

    private _getElapsedTime(): string {
        if (!this.toolCall.endTime) {return '';}
        const elapsed = this.toolCall.endTime - this.toolCall.startTime;
        return elapsed < 1000 ? '<1s' : `${Math.floor(elapsed / 1000)}s`;
    }

    private _getIcon(): vscode.ThemeIcon {
        // Icon based on tool name
        const name = this.toolCall.name.toLowerCase();
        if (name.includes('read') || name.includes('file')) {
            return new vscode.ThemeIcon('file');
        }
        if (name.includes('search') || name.includes('find')) {
            return new vscode.ThemeIcon('search');
        }
        if (name.includes('edit') || name.includes('write')) {
            return new vscode.ThemeIcon('edit');
        }
        if (name.includes('bash') || name.includes('shell') || name.includes('terminal')) {
            return new vscode.ThemeIcon('terminal');
        }
        if (name.includes('git')) {
            return new vscode.ThemeIcon('git-commit');
        }
        return new vscode.ThemeIcon('wrench');
    }
}

/**
 * Tree item for agent info/metadata
 */
class AgentInfoItem extends vscode.TreeItem {
    constructor(label: string, value: string, icon?: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = value;
        this.iconPath = icon ? new vscode.ThemeIcon(icon) : undefined;
        this.contextValue = 'agent-info';
    }
}

/**
 * Agents View TreeDataProvider
 */
export class AgentsViewProvider implements vscode.TreeDataProvider<AgentTreeItem>, vscode.Disposable {
    private _onDidChangeTreeData: vscode.EventEmitter<AgentTreeItem | undefined | null | void> =
        new vscode.EventEmitter<AgentTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<AgentTreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private _agents: Map<string, AgentTask> = new Map();
    private _disposables: vscode.Disposable[] = [];
    private _refreshInterval: NodeJS.Timeout | null = null;
    private _isLoading: boolean = false;
    private _lastFetchTime: number = 0;
    private _statusBarItem: vscode.StatusBarItem;

    constructor(
        private readonly _client: VictorClient,
        private readonly _outputChannel?: vscode.OutputChannel
    ) {
        // Create status bar item for active agents
        this._statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            98
        );
        this._statusBarItem.command = 'victor.startAgent';
        this._updateStatusBar();
        this._disposables.push(this._statusBarItem);

        // Subscribe to WebSocket agent events
        this._subscribeToAgentEvents();

        // Initial fetch from backend
        this._fetchAgentsFromBackend();

        // Start auto-refresh for running agents
        this._startAutoRefresh();
    }

    private _updateStatusBar(): void {
        const activeCount = this.getActiveCount();
        if (activeCount > 0) {
            this._statusBarItem.text = `$(robot) ${activeCount} agent${activeCount > 1 ? 's' : ''}`;
            this._statusBarItem.tooltip = `${activeCount} background agent${activeCount > 1 ? 's' : ''} running. Click to start new agent.`;
            this._statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            this._statusBarItem.show();
        } else {
            this._statusBarItem.text = '$(robot) Agents';
            this._statusBarItem.tooltip = 'Start a background agent';
            this._statusBarItem.backgroundColor = undefined;
            this._statusBarItem.show();
        }
    }

    private _shouldShowNotifications(): boolean {
        const config = vscode.workspace.getConfiguration('victor');
        return config.get('agents.showNotifications', true);
    }

    /**
     * Subscribe to WebSocket events for real-time updates
     */
    private _subscribeToAgentEvents(): void {
        // Listen for agent events via the client's event handler
        const eventHandler = (event: AgentEventData) => {
            switch (event.type) {
                case 'agent_started':
                    this._handleAgentStarted(event.data as AgentTask);
                    break;
                case 'agent_running':
                    this._handleAgentUpdate(event.data as Partial<AgentTask>);
                    break;
                case 'agent_tool_call':
                    this._handleToolCall(event.data as AgentToolCallEvent['data']);
                    break;
                case 'agent_tool_result':
                    this._handleToolResult(event.data as AgentToolResultEvent['data']);
                    break;
                case 'agent_completed':
                case 'agent_error':
                case 'agent_cancelled':
                    this._handleAgentUpdate(event.data as Partial<AgentTask>);
                    break;
            }
        };

        // Register the event handler with the client
        this._client.onAgentEvent(eventHandler as (event: { type: string; data: unknown }) => void);
        this._log('Subscribed to agent events');
    }

    private _handleAgentStarted(data: AgentTask): void {
        const task = this._convertBackendAgent(data);
        this._agents.set(task.id, task);
        this.refresh();
        this._updateStatusBar();
        this._log(`Agent started: ${task.id} - ${task.name}`);
    }

    private _handleAgentUpdate(data: Partial<AgentTask>): void {
        const task = this._convertBackendAgent(data);
        const previousTask = this._agents.get(task.id);
        this._agents.set(task.id, task);
        this.refresh();
        this._updateStatusBar();
        this._log(`Agent updated: ${task.id} - ${task.status}`);

        // Show notifications for status changes (if enabled)
        if (this._shouldShowNotifications() && previousTask && previousTask.status !== task.status) {
            switch (task.status) {
                case AgentStatus.Completed:
                    vscode.window.showInformationMessage(
                        `Agent completed: ${task.name}`,
                        'View Output'
                    ).then(action => {
                        if (action === 'View Output') {
                            vscode.commands.executeCommand('victor.viewAgentOutput', { task });
                        }
                    });
                    break;
                case AgentStatus.Error:
                    vscode.window.showErrorMessage(
                        `Agent failed: ${task.name}${task.error ? ` - ${task.error}` : ''}`,
                        'View Details'
                    ).then(action => {
                        if (action === 'View Details') {
                            vscode.commands.executeCommand('victor.viewAgentOutput', { task });
                        }
                    });
                    break;
            }
        }
    }

    private _handleToolCall(data: AgentToolCallEvent['data']): void {
        const agentId = data.agent_id;
        const agent = this._agents.get(agentId);
        if (agent && data.tool_call) {
            const toolCall: AgentToolCall = {
                id: data.tool_call.id,
                name: data.tool_call.name,
                status: data.tool_call.status,
                startTime: Date.now(),
            };
            agent.toolCalls.push(toolCall);
            this.refresh();
        }
    }

    private _handleToolResult(data: AgentToolResultEvent['data']): void {
        const agentId = data.agent_id;
        const toolCallId = data.tool_call_id;
        const agent = this._agents.get(agentId);
        if (agent) {
            const toolCall = agent.toolCalls.find(tc => tc.id === toolCallId);
            if (toolCall) {
                toolCall.status = data.status || 'success';
                toolCall.endTime = Date.now();
                this.refresh();
            }
        }
    }

    /**
     * Convert backend agent format to local AgentTask
     */
    private _convertBackendAgent(data: BackendAgentData | Partial<BackendAgentData>): AgentTask {
        return {
            id: data.id || 'unknown',
            name: data.name || (data.task ? data.task.slice(0, 40) : 'Agent'),
            description: data.description || data.task || '',
            status: data.status as AgentStatus,
            progress: data.progress,
            startTime: (data.start_time ?? Date.now() / 1000) * 1000, // Convert to ms
            endTime: data.end_time ? data.end_time * 1000 : undefined,
            mode: data.mode,
            output: data.output,
            error: data.error,
            toolCalls: (data.tool_calls || []).map((tc: BackendToolCallData) => ({
                id: tc.id,
                name: tc.name,
                status: tc.status as AgentToolCall['status'],
                startTime: tc.start_time * 1000,
                endTime: tc.end_time ? tc.end_time * 1000 : undefined,
                result: tc.result,
            })),
        };
    }

    /**
     * Fetch agents from backend
     */
    async _fetchAgentsFromBackend(): Promise<void> {
        // Debounce: don't fetch more than once per second
        const now = Date.now();
        if (now - this._lastFetchTime < 1000) {
            return;
        }
        this._lastFetchTime = now;

        if (this._isLoading) {return;}
        this._isLoading = true;

        try {
            const agents = await this._client.listAgents();

            // Update local state
            this._agents.clear();
            for (const agentData of agents) {
                const task = this._convertBackendAgent(agentData);
                this._agents.set(task.id, task);
            }

            this.refresh();
            this._updateStatusBar();
            this._log(`Fetched ${agents.length} agents from backend`);
        } catch (error) {
            this._log(`Failed to fetch agents: ${error}`);
        } finally {
            this._isLoading = false;
        }
    }

    dispose(): void {
        if (this._refreshInterval) {
            clearInterval(this._refreshInterval);
        }
        this._disposables.forEach(d => d.dispose());
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    // =========================================================================
    // TreeDataProvider Implementation
    // =========================================================================

    getTreeItem(element: AgentTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: AgentTreeItem): Promise<AgentTreeItem[]> {
        if (!element) {
            // Root level: show all agents
            return this._getAgentItems();
        }

        if (element instanceof AgentTaskItem) {
            // Show tool calls for this agent
            return this._getToolCallItems(element.task);
        }

        return [];
    }

    getParent(element: AgentTreeItem): vscode.ProviderResult<AgentTreeItem> {
        if (element instanceof AgentToolCallItem) {
            const agent = this._agents.get(element.parentId);
            if (agent) {
                return new AgentTaskItem(agent, vscode.TreeItemCollapsibleState.Expanded);
            }
        }
        return null;
    }

    // =========================================================================
    // Agent Management
    // =========================================================================

    /**
     * Add or update an agent task
     */
    addAgent(task: AgentTask): void {
        this._agents.set(task.id, task);
        this.refresh();
        this._log(`Agent added: ${task.id} - ${task.name}`);
    }

    /**
     * Update an existing agent
     */
    updateAgent(id: string, updates: Partial<AgentTask>): void {
        const agent = this._agents.get(id);
        if (agent) {
            Object.assign(agent, updates);
            this.refresh();
        }
    }

    /**
     * Add a tool call to an agent
     */
    addToolCall(agentId: string, toolCall: AgentToolCall): void {
        const agent = this._agents.get(agentId);
        if (agent) {
            agent.toolCalls.push(toolCall);
            this.refresh();
        }
    }

    /**
     * Update a tool call status
     */
    updateToolCall(agentId: string, toolCallId: string, updates: Partial<AgentToolCall>): void {
        const agent = this._agents.get(agentId);
        if (agent) {
            const toolCall = agent.toolCalls.find(tc => tc.id === toolCallId);
            if (toolCall) {
                Object.assign(toolCall, updates);
                this.refresh();
            }
        }
    }

    /**
     * Remove an agent
     */
    removeAgent(id: string): void {
        this._agents.delete(id);
        this.refresh();
    }

    /**
     * Clear all completed/failed agents
     */
    async clearCompleted(): Promise<void> {
        try {
            // Clear on backend
            await this._client.clearAgents();

            // Clear locally
            for (const [id, agent] of this._agents.entries()) {
                if (agent.status === AgentStatus.Completed ||
                    agent.status === AgentStatus.Error ||
                    agent.status === AgentStatus.Cancelled) {
                    this._agents.delete(id);
                }
            }
            this.refresh();
            this._log('Cleared completed agents');
        } catch (error) {
            this._log(`Failed to clear agents: ${error}`);
        }
    }

    /**
     * Cancel a running agent
     */
    async cancelAgent(id: string): Promise<void> {
        const agent = this._agents.get(id);
        if (agent && (agent.status === AgentStatus.Running || agent.status === AgentStatus.Pending)) {
            try {
                // Cancel on backend
                await this._client.cancelAgent(id);

                // Update locally
                agent.status = AgentStatus.Cancelled;
                agent.endTime = Date.now();
                this.refresh();
                this._log(`Cancelled agent: ${id}`);
            } catch (error) {
                this._log(`Failed to cancel agent: ${error}`);
            }
        }
    }

    /**
     * Start a new background agent
     */
    async startAgent(task: string, mode: 'build' | 'plan' | 'explore' = 'build'): Promise<string | null> {
        // Check max concurrent agents limit
        const config = vscode.workspace.getConfiguration('victor.agents');
        const maxConcurrent = config.get<number>('maxConcurrent', 4);
        const activeCount = this.getActiveCount();

        if (activeCount >= maxConcurrent) {
            const message = `Maximum concurrent agents (${maxConcurrent}) reached. Wait for an agent to complete or cancel one.`;
            this._log(message);
            vscode.window.showWarningMessage(message);
            return null;
        }

        try {
            const agentId = await this._client.startAgent(task, mode);
            this._log(`Started agent: ${agentId}`);
            // The WebSocket event will update the UI
            return agentId;
        } catch (error) {
            this._log(`Failed to start agent: ${error}`);
            vscode.window.showErrorMessage(`Failed to start agent: ${error}`);
            return null;
        }
    }

    /**
     * Get count of active agents
     */
    getActiveCount(): number {
        let count = 0;
        for (const agent of this._agents.values()) {
            if (agent.status === AgentStatus.Running || agent.status === AgentStatus.Pending) {
                count++;
            }
        }
        return count;
    }

    // =========================================================================
    // Private Methods
    // =========================================================================

    private _getAgentItems(): AgentTreeItem[] {
        if (this._agents.size === 0) {
            return [new AgentInfoItem('No active agents', 'Start a task to see agents here', 'info')];
        }

        const items: AgentTaskItem[] = [];

        // Sort: running first, then pending, then by start time
        const sorted = Array.from(this._agents.values()).sort((a, b) => {
            const statusOrder = {
                [AgentStatus.Running]: 0,
                [AgentStatus.Pending]: 1,
                [AgentStatus.Paused]: 2,
                [AgentStatus.Completed]: 3,
                [AgentStatus.Error]: 4,
                [AgentStatus.Cancelled]: 5,
            };
            const statusDiff = statusOrder[a.status] - statusOrder[b.status];
            if (statusDiff !== 0) {return statusDiff;}
            return b.startTime - a.startTime;  // Newer first
        });

        for (const agent of sorted) {
            const hasChildren = agent.toolCalls.length > 0;
            items.push(new AgentTaskItem(
                agent,
                hasChildren
                    ? vscode.TreeItemCollapsibleState.Collapsed
                    : vscode.TreeItemCollapsibleState.None
            ));
        }

        return items;
    }

    private _getToolCallItems(task: AgentTask): AgentTreeItem[] {
        return task.toolCalls.map(tc => new AgentToolCallItem(tc, task.id));
    }

    private _startAutoRefresh(): void {
        // Refresh every 2 seconds while there are running agents
        this._refreshInterval = setInterval(() => {
            const hasRunning = Array.from(this._agents.values()).some(
                a => a.status === AgentStatus.Running
            );
            if (hasRunning) {
                this.refresh();
            }
        }, 2000);
    }

    private _log(message: string): void {
        this._outputChannel?.appendLine(`[Agents] ${message}`);
    }
}

/**
 * Register agent-related commands
 */
export function registerAgentCommands(
    context: vscode.ExtensionContext,
    agentsView: AgentsViewProvider
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshAgents', async () => {
            await agentsView._fetchAgentsFromBackend();
        }),
        vscode.commands.registerCommand('victor.clearAgents', async () => {
            await agentsView.clearCompleted();
            vscode.window.showInformationMessage('Cleared completed agents');
        }),
        vscode.commands.registerCommand('victor.cancelAgent', async (item: AgentTaskItem) => {
            if (item?.task?.id) {
                await agentsView.cancelAgent(item.task.id);
                vscode.window.showInformationMessage(`Cancelled agent: ${item.task.name}`);
            }
        }),
        vscode.commands.registerCommand('victor.startAgent', async () => {
            // Check for selection to provide context
            const editor = vscode.window.activeTextEditor;
            let contextInfo = '';
            if (editor && !editor.selection.isEmpty) {
                const selectedText = editor.document.getText(editor.selection);
                const relativePath = vscode.workspace.asRelativePath(editor.document.uri);
                const startLine = editor.selection.start.line + 1;
                const endLine = editor.selection.end.line + 1;
                contextInfo = `\n\nContext from ${relativePath}:${startLine}-${endLine}:\n\`\`\`${editor.document.languageId}\n${selectedText}\n\`\`\``;
            }

            // Get task from user
            const task = await vscode.window.showInputBox({
                prompt: contextInfo ? 'Enter the task (selected code will be included as context)' : 'Enter the task for the background agent',
                placeHolder: 'e.g., Refactor the authentication module',
                validateInput: (value) => {
                    if (!value || value.trim().length < 5) {
                        return 'Please enter a task description (at least 5 characters)';
                    }
                    return null;
                }
            });

            if (!task) {return;}

            // Get mode from user
            const modeOption = await vscode.window.showQuickPick([
                { label: 'Build', description: 'Execute changes and modifications', value: 'build' as const },
                { label: 'Plan', description: 'Create a detailed plan without executing', value: 'plan' as const },
                { label: 'Explore', description: 'Investigate and analyze codebase', value: 'explore' as const },
            ], {
                placeHolder: 'Select agent mode',
            });

            if (!modeOption) {return;}

            // Combine task with context
            const fullTask = task + contextInfo;

            // Start the agent
            const agentId = await agentsView.startAgent(fullTask, modeOption.value);
            if (agentId) {
                vscode.window.showInformationMessage(`Started agent: ${agentId}`);
            }
        }),
        vscode.commands.registerCommand('victor.viewAgentOutput', async (item: AgentTaskItem) => {
            if (!item?.task) {return;}

            const agent = item.task;
            const content = [
                `# Agent: ${agent.name}`,
                `**Status:** ${agent.status}`,
                `**Mode:** ${agent.mode || 'build'}`,
                `**Progress:** ${agent.progress || 0}%`,
                '',
                '## Description',
                agent.description,
                '',
            ];

            if (agent.output) {
                content.push('## Output', '```', agent.output, '```', '');
            }

            if (agent.error) {
                content.push('## Error', '```', agent.error, '```', '');
            }

            if (agent.toolCalls.length > 0) {
                content.push('## Tool Calls');
                for (const tc of agent.toolCalls) {
                    content.push(`- **${tc.name}** (${tc.status})`);
                    if (tc.result) {
                        content.push(`  ${tc.result.slice(0, 100)}...`);
                    }
                }
            }

            // Show in output channel or as markdown preview
            const doc = await vscode.workspace.openTextDocument({
                content: content.join('\n'),
                language: 'markdown',
            });
            await vscode.window.showTextDocument(doc, { preview: true });
        })
    );
}

/**
 * Create demo agent for testing
 */
export function createDemoAgent(): AgentTask {
    return {
        id: `agent-${Date.now()}`,
        name: 'Code Review',
        description: 'Reviewing selected code for issues and improvements',
        status: AgentStatus.Running,
        progress: 45,
        startTime: Date.now() - 30000,
        mode: 'explore',
        toolCalls: [
            {
                id: 'tc-1',
                name: 'read_file',
                status: 'success',
                startTime: Date.now() - 25000,
                endTime: Date.now() - 24000,
                result: 'Read 150 lines',
            },
            {
                id: 'tc-2',
                name: 'code_search',
                status: 'success',
                startTime: Date.now() - 23000,
                endTime: Date.now() - 20000,
                result: 'Found 12 references',
            },
            {
                id: 'tc-3',
                name: 'analyze_code',
                status: 'running',
                startTime: Date.now() - 5000,
            },
        ],
    };
}
