/**
 * Workflows View Provider
 *
 * Provides a TreeView for browsing and executing workflow templates.
 * Displays available workflows, their steps, and execution history.
 *
 * Features:
 * - Browse workflow templates
 * - View workflow steps and configuration
 * - Execute workflows with parameters
 * - Track workflow execution progress
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

// Workflow status
export enum WorkflowStatus {
    Available = 'available',
    Running = 'running',
    Completed = 'completed',
    Failed = 'failed',
    Cancelled = 'cancelled',
}

// Workflow step type
export interface WorkflowStep {
    id: string;
    name: string;
    type: 'agent' | 'condition' | 'parallel' | 'loop';
    role?: string;
    goal?: string;
    status?: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
    duration?: number;
    output?: string;
}

// Workflow template
export interface WorkflowTemplate {
    id: string;
    name: string;
    description: string;
    category: string;
    steps: WorkflowStep[];
    parameters?: WorkflowParameter[];
    estimatedDuration?: string;
    tags?: string[];
}

// Workflow parameter
export interface WorkflowParameter {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'file' | 'selection';
    description: string;
    required: boolean;
    default?: string | number | boolean;
    options?: string[];  // For selection type
}

// Workflow execution
export interface WorkflowExecution {
    id: string;
    workflowId: string;
    workflowName: string;
    status: WorkflowStatus;
    parameters: Record<string, unknown>;
    currentStep?: string;
    progress: number;
    startTime: number;
    endTime?: number;
    steps: WorkflowStep[];
    output?: string;
    error?: string;
}

// Tree item types
type WorkflowTreeItem = WorkflowCategoryItem | WorkflowTemplateItem | WorkflowStepItem |
    WorkflowExecutionItem | WorkflowInfoItem;

/**
 * Tree item for workflow category
 */
class WorkflowCategoryItem extends vscode.TreeItem {
    constructor(
        public readonly category: string,
        public readonly workflows: WorkflowTemplate[]
    ) {
        super(category, vscode.TreeItemCollapsibleState.Collapsed);

        this.description = `${workflows.length} workflow${workflows.length !== 1 ? 's' : ''}`;
        this.iconPath = this._getIcon();
        this.contextValue = 'workflow-category';
    }

    private _getIcon(): vscode.ThemeIcon {
        const categoryIcons: Record<string, string> = {
            'Code Review': 'eye',
            'Testing': 'beaker',
            'Refactoring': 'edit',
            'Documentation': 'book',
            'Security': 'shield',
            'Performance': 'zap',
            'Debugging': 'bug',
            'General': 'workflow',
        };

        const icon = categoryIcons[this.category] || 'folder';
        return new vscode.ThemeIcon(icon);
    }
}

/**
 * Tree item for workflow template
 */
class WorkflowTemplateItem extends vscode.TreeItem {
    constructor(
        public readonly workflow: WorkflowTemplate
    ) {
        super(workflow.name, vscode.TreeItemCollapsibleState.Collapsed);

        this.description = workflow.description.slice(0, 50) + (workflow.description.length > 50 ? '...' : '');
        this.tooltip = this._getTooltip();
        this.iconPath = new vscode.ThemeIcon('workflow', new vscode.ThemeColor('charts.blue'));
        this.contextValue = 'workflow-template';
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### ${this.workflow.name}\n\n`);
        md.appendMarkdown(`${this.workflow.description}\n\n`);
        md.appendMarkdown(`**Steps:** ${this.workflow.steps.length}\n\n`);

        if (this.workflow.estimatedDuration) {
            md.appendMarkdown(`**Estimated Duration:** ${this.workflow.estimatedDuration}\n\n`);
        }

        if (this.workflow.tags && this.workflow.tags.length > 0) {
            md.appendMarkdown(`**Tags:** ${this.workflow.tags.join(', ')}\n\n`);
        }

        if (this.workflow.parameters && this.workflow.parameters.length > 0) {
            md.appendMarkdown(`**Parameters:**\n`);
            for (const param of this.workflow.parameters) {
                const required = param.required ? ' (required)' : '';
                md.appendMarkdown(`- \`${param.name}\`${required}: ${param.description}\n`);
            }
        }

        return md;
    }
}

/**
 * Tree item for workflow step
 */
class WorkflowStepItem extends vscode.TreeItem {
    constructor(
        public readonly step: WorkflowStep,
        public readonly workflowId: string,
        public readonly index: number
    ) {
        super(`${index + 1}. ${step.name}`, vscode.TreeItemCollapsibleState.None);

        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = `workflow-step-${step.status || 'pending'}`;
    }

    private _getDescription(): string {
        const typeLabel = this.step.type.charAt(0).toUpperCase() + this.step.type.slice(1);
        if (this.step.role) {
            return `${typeLabel} (${this.step.role})`;
        }
        return typeLabel;
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### Step ${this.index + 1}: ${this.step.name}\n\n`);
        md.appendMarkdown(`**Type:** ${this.step.type}\n\n`);

        if (this.step.role) {
            md.appendMarkdown(`**Role:** ${this.step.role}\n\n`);
        }

        if (this.step.goal) {
            md.appendMarkdown(`**Goal:** ${this.step.goal}\n\n`);
        }

        if (this.step.status) {
            md.appendMarkdown(`**Status:** ${this.step.status}\n\n`);
        }

        if (this.step.duration) {
            md.appendMarkdown(`**Duration:** ${this.step.duration}s\n\n`);
        }

        return md;
    }

    private _getIcon(): vscode.ThemeIcon {
        const typeIcons: Record<string, string> = {
            'agent': 'robot',
            'condition': 'question',
            'parallel': 'split-horizontal',
            'loop': 'sync',
        };

        const icon = typeIcons[this.step.type] || 'circle';

        switch (this.step.status) {
            case 'running':
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.blue'));
            case 'completed':
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.green'));
            case 'failed':
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.red'));
            case 'skipped':
                return new vscode.ThemeIcon(icon, new vscode.ThemeColor('charts.gray'));
            default:
                return new vscode.ThemeIcon(icon);
        }
    }
}

/**
 * Tree item for workflow execution
 */
class WorkflowExecutionItem extends vscode.TreeItem {
    constructor(
        public readonly execution: WorkflowExecution
    ) {
        super(execution.workflowName, vscode.TreeItemCollapsibleState.Collapsed);

        this.id = execution.id;
        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = `workflow-execution-${execution.status}`;
    }

    private _getDescription(): string {
        const progress = `${this.execution.progress}%`;
        const elapsed = this._getElapsedTime();
        return `${this._getStatusEmoji()} ${progress} • ${elapsed}`;
    }

    private _getStatusEmoji(): string {
        switch (this.execution.status) {
            case WorkflowStatus.Running: return '🔄';
            case WorkflowStatus.Completed: return '✅';
            case WorkflowStatus.Failed: return '❌';
            case WorkflowStatus.Cancelled: return '🚫';
            default: return '⏳';
        }
    }

    private _getElapsedTime(): string {
        const end = this.execution.endTime || Date.now();
        const elapsed = end - this.execution.startTime;

        if (elapsed < 1000) { return '<1s'; }
        if (elapsed < 60000) { return `${Math.floor(elapsed / 1000)}s`; }
        if (elapsed < 3600000) { return `${Math.floor(elapsed / 60000)}m`; }
        return `${Math.floor(elapsed / 3600000)}h`;
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### ${this.execution.workflowName}\n\n`);
        md.appendMarkdown(`**Status:** ${this.execution.status}\n\n`);
        md.appendMarkdown(`**Progress:** ${this.execution.progress}%\n\n`);

        if (this.execution.currentStep) {
            md.appendMarkdown(`**Current Step:** ${this.execution.currentStep}\n\n`);
        }

        if (this.execution.error) {
            md.appendMarkdown(`**Error:** ${this.execution.error}\n\n`);
        }

        return md;
    }

    private _getIcon(): vscode.ThemeIcon {
        switch (this.execution.status) {
            case WorkflowStatus.Running:
                return new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('charts.blue'));
            case WorkflowStatus.Completed:
                return new vscode.ThemeIcon('pass', new vscode.ThemeColor('charts.green'));
            case WorkflowStatus.Failed:
                return new vscode.ThemeIcon('error', new vscode.ThemeColor('charts.red'));
            case WorkflowStatus.Cancelled:
                return new vscode.ThemeIcon('circle-slash', new vscode.ThemeColor('charts.gray'));
            default:
                return new vscode.ThemeIcon('workflow');
        }
    }
}

/**
 * Tree item for info/placeholder
 */
class WorkflowInfoItem extends vscode.TreeItem {
    constructor(label: string, description: string, icon?: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = description;
        this.iconPath = icon ? new vscode.ThemeIcon(icon) : undefined;
        this.contextValue = 'workflow-info';
    }
}

/**
 * Workflows View TreeDataProvider
 */
export class WorkflowsViewProvider implements vscode.TreeDataProvider<WorkflowTreeItem>, vscode.Disposable {
    private _onDidChangeTreeData: vscode.EventEmitter<WorkflowTreeItem | undefined | null | void> =
        new vscode.EventEmitter<WorkflowTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<WorkflowTreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private _templates: WorkflowTemplate[] = [];
    private _executions: Map<string, WorkflowExecution> = new Map();
    private _disposables: vscode.Disposable[] = [];
    private _refreshInterval: NodeJS.Timeout | null = null;
    private _showExecutions: boolean = false;  // Toggle between templates and executions

    constructor(
        private readonly _client: VictorClient,
        private readonly _outputChannel?: vscode.OutputChannel
    ) {
        // Subscribe to workflow events
        this._subscribeToWorkflowEvents();

        // Initial fetch
        this._fetchWorkflowsFromBackend();

        // Start auto-refresh
        this._startAutoRefresh();
    }

    private _subscribeToWorkflowEvents(): void {
        this._client.onAgentEvent((event) => {
            switch (event.type) {
                case 'workflow_started':
                    this._handleWorkflowStarted(event.data);
                    break;
                case 'workflow_progress':
                    this._handleWorkflowProgress(event.data);
                    break;
                case 'workflow_step_completed':
                    this._handleStepCompleted(event.data);
                    break;
                case 'workflow_completed':
                case 'workflow_failed':
                case 'workflow_cancelled':
                    this._handleWorkflowCompleted(event.data);
                    break;
            }
        });

        this._log('Subscribed to workflow events');
    }

    private _handleWorkflowStarted(data: unknown): void {
        const execution = this._convertExecution(data);
        this._executions.set(execution.id, execution);
        this.refresh();
        this._log(`Workflow started: ${execution.id}`);
    }

    private _handleWorkflowProgress(data: unknown): void {
        const d = data as { execution_id: string; progress: number; current_step?: string };
        const execution = this._executions.get(d.execution_id);
        if (execution) {
            execution.progress = d.progress;
            execution.currentStep = d.current_step;
            this.refresh();
        }
    }

    private _handleStepCompleted(data: unknown): void {
        const d = data as { execution_id: string; step_id: string; status: string; duration?: number };
        const execution = this._executions.get(d.execution_id);
        if (execution) {
            const step = execution.steps.find(s => s.id === d.step_id);
            if (step) {
                step.status = d.status as 'completed' | 'failed' | 'skipped';
                step.duration = d.duration;
                this.refresh();
            }
        }
    }

    private _handleWorkflowCompleted(data: unknown): void {
        const execution = this._convertExecution(data);
        this._executions.set(execution.id, execution);
        this.refresh();
        this._log(`Workflow completed: ${execution.id} (${execution.status})`);

        // Show notification
        if (execution.status === WorkflowStatus.Completed) {
            vscode.window.showInformationMessage(`Workflow completed: ${execution.workflowName}`);
        } else if (execution.status === WorkflowStatus.Failed) {
            vscode.window.showErrorMessage(`Workflow failed: ${execution.workflowName}`);
        }
    }

    private _convertExecution(data: unknown): WorkflowExecution {
        const d = data as Record<string, unknown>;
        return {
            id: d.id as string,
            workflowId: d.workflow_id as string,
            workflowName: d.workflow_name as string,
            status: d.status as WorkflowStatus,
            parameters: (d.parameters as Record<string, unknown>) || {},
            currentStep: d.current_step as string | undefined,
            progress: (d.progress as number) || 0,
            startTime: ((d.start_time as number) || Date.now() / 1000) * 1000,
            endTime: d.end_time ? (d.end_time as number) * 1000 : undefined,
            steps: (d.steps as WorkflowStep[]) || [],
            output: d.output as string | undefined,
            error: d.error as string | undefined,
        };
    }

    async _fetchWorkflowsFromBackend(): Promise<void> {
        try {
            const templates = await this._client.listWorkflowTemplates();
            this._templates = templates.map(t => ({
                ...t,
                steps: t.steps.map(s => ({
                    ...s,
                    type: s.type as 'agent' | 'condition' | 'parallel' | 'loop',
                })),
                parameters: t.parameters?.map(p => ({
                    ...p,
                    type: p.type as 'string' | 'number' | 'boolean' | 'file' | 'selection',
                    default: p.default as string | number | boolean | undefined,
                })),
            }));

            const executions = await this._client.listWorkflowExecutions();
            this._executions.clear();
            for (const exec of executions) {
                const execution = this._convertExecution(exec);
                this._executions.set(execution.id, execution);
            }

            this.refresh();
            this._log(`Fetched ${templates.length} templates, ${executions.length} executions`);
        } catch (error) {
            this._log(`Failed to fetch workflows: ${error}`);
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

    toggleView(): void {
        this._showExecutions = !this._showExecutions;
        this.refresh();
    }

    // TreeDataProvider implementation
    getTreeItem(element: WorkflowTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: WorkflowTreeItem): Promise<WorkflowTreeItem[]> {
        if (!element) {
            return this._showExecutions ? this._getExecutionItems() : this._getTemplateItems();
        }

        if (element instanceof WorkflowCategoryItem) {
            return element.workflows.map(w => new WorkflowTemplateItem(w));
        }

        if (element instanceof WorkflowTemplateItem) {
            return element.workflow.steps.map((step, i) =>
                new WorkflowStepItem(step, element.workflow.id, i)
            );
        }

        if (element instanceof WorkflowExecutionItem) {
            return element.execution.steps.map((step, i) =>
                new WorkflowStepItem(step, element.execution.id, i)
            );
        }

        return [];
    }

    getParent(_element: WorkflowTreeItem): vscode.ProviderResult<WorkflowTreeItem> {
        return null;
    }

    // Workflow execution methods
    async executeWorkflow(templateId: string, parameters: Record<string, unknown>): Promise<string | null> {
        try {
            const executionId = await this._client.executeWorkflow(templateId, parameters);
            this._log(`Started workflow execution: ${executionId}`);
            return executionId;
        } catch (error) {
            this._log(`Failed to execute workflow: ${error}`);
            vscode.window.showErrorMessage(`Failed to execute workflow: ${error}`);
            return null;
        }
    }

    async cancelExecution(executionId: string): Promise<void> {
        try {
            await this._client.cancelWorkflowExecution(executionId);
            this._log(`Cancelled workflow: ${executionId}`);
        } catch (error) {
            this._log(`Failed to cancel workflow: ${error}`);
            vscode.window.showErrorMessage(`Failed to cancel workflow: ${error}`);
        }
    }

    private _getTemplateItems(): WorkflowTreeItem[] {
        if (this._templates.length === 0) {
            return [new WorkflowInfoItem('No workflows', 'No workflow templates available', 'info')];
        }

        // Group by category
        const categories = new Map<string, WorkflowTemplate[]>();
        for (const template of this._templates) {
            const category = template.category || 'General';
            const list = categories.get(category) || [];
            list.push(template);
            categories.set(category, list);
        }

        // Sort categories and return
        return Array.from(categories.entries())
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([category, workflows]) => new WorkflowCategoryItem(category, workflows));
    }

    private _getExecutionItems(): WorkflowTreeItem[] {
        if (this._executions.size === 0) {
            return [new WorkflowInfoItem('No executions', 'Execute a workflow to see history', 'info')];
        }

        // Sort by start time (newest first)
        const sorted = Array.from(this._executions.values())
            .sort((a, b) => b.startTime - a.startTime);

        return sorted.map(exec => new WorkflowExecutionItem(exec));
    }

    private _startAutoRefresh(): void {
        this._refreshInterval = setInterval(() => {
            const hasRunning = Array.from(this._executions.values()).some(
                e => e.status === WorkflowStatus.Running
            );
            if (hasRunning) {
                this.refresh();
            }
        }, 2000);
    }

    private _log(message: string): void {
        this._outputChannel?.appendLine(`[Workflows] ${message}`);
    }
}

/**
 * Register workflow-related commands
 */
export function registerWorkflowCommands(
    context: vscode.ExtensionContext,
    workflowsView: WorkflowsViewProvider
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshWorkflows', async () => {
            await workflowsView._fetchWorkflowsFromBackend();
        }),

        vscode.commands.registerCommand('victor.toggleWorkflowView', () => {
            workflowsView.toggleView();
        }),

        vscode.commands.registerCommand('victor.executeWorkflow', async (item: WorkflowTemplateItem) => {
            if (!item?.workflow) { return; }

            const workflow = item.workflow;

            // Collect parameters if any
            const parameters: Record<string, unknown> = {};

            if (workflow.parameters && workflow.parameters.length > 0) {
                for (const param of workflow.parameters) {
                    let value: string | undefined;

                    if (param.type === 'selection' && param.options) {
                        const selected = await vscode.window.showQuickPick(param.options, {
                            placeHolder: param.description,
                        });
                        if (!selected && param.required) {
                            vscode.window.showWarningMessage('Workflow execution cancelled');
                            return;
                        }
                        value = selected;
                    } else if (param.type === 'file') {
                        const files = await vscode.window.showOpenDialog({
                            canSelectFiles: true,
                            canSelectFolders: false,
                            title: param.description,
                        });
                        if ((!files || files.length === 0) && param.required) {
                            vscode.window.showWarningMessage('Workflow execution cancelled');
                            return;
                        }
                        value = files?.[0]?.fsPath;
                    } else if (param.type === 'boolean') {
                        const selected = await vscode.window.showQuickPick(['Yes', 'No'], {
                            placeHolder: param.description,
                        });
                        if (!selected && param.required) {
                            vscode.window.showWarningMessage('Workflow execution cancelled');
                            return;
                        }
                        parameters[param.name] = selected === 'Yes';
                        continue;
                    } else {
                        value = await vscode.window.showInputBox({
                            prompt: param.description,
                            value: param.default?.toString(),
                            validateInput: (v) => {
                                if (param.required && !v) {
                                    return 'This parameter is required';
                                }
                                if (param.type === 'number' && v && isNaN(Number(v))) {
                                    return 'Please enter a valid number';
                                }
                                return null;
                            },
                        });

                        if (!value && param.required) {
                            vscode.window.showWarningMessage('Workflow execution cancelled');
                            return;
                        }
                    }

                    if (value !== undefined) {
                        parameters[param.name] = param.type === 'number' ? Number(value) : value;
                    }
                }
            }

            // Execute workflow
            const executionId = await workflowsView.executeWorkflow(workflow.id, parameters);
            if (executionId) {
                vscode.window.showInformationMessage(`Started workflow: ${workflow.name}`);
            }
        }),

        vscode.commands.registerCommand('victor.cancelWorkflow', async (item: WorkflowExecutionItem) => {
            if (item?.execution?.id) {
                await workflowsView.cancelExecution(item.execution.id);
                vscode.window.showInformationMessage('Workflow cancelled');
            }
        }),

        vscode.commands.registerCommand('victor.viewWorkflowOutput', async (item: WorkflowExecutionItem) => {
            if (!item?.execution) { return; }

            const execution = item.execution;
            const content = [
                `# Workflow: ${execution.workflowName}`,
                `**Status:** ${execution.status}`,
                `**Progress:** ${execution.progress}%`,
                '',
                '## Steps',
            ];

            for (const step of execution.steps) {
                const statusEmoji = {
                    'pending': '⏳',
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌',
                    'skipped': '⏭️',
                }[step.status || 'pending'];

                content.push(`${statusEmoji} **${step.name}** (${step.type})`);
                if (step.duration) {
                    content.push(`   Duration: ${step.duration}s`);
                }
            }

            if (execution.output) {
                content.push('', '## Output', '```', execution.output, '```');
            }

            if (execution.error) {
                content.push('', '## Error', '```', execution.error, '```');
            }

            const doc = await vscode.workspace.openTextDocument({
                content: content.join('\n'),
                language: 'markdown',
            });
            await vscode.window.showTextDocument(doc, { preview: true });
        })
    );
}
