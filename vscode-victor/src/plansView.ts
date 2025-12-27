/**
 * Plans View Provider
 *
 * Provides a TreeView for viewing and managing execution plans.
 * Displays plan steps, their status, and allows plan approval/modification.
 *
 * Plans are created during PLAN mode and can be:
 * - Viewed before execution
 * - Approved for BUILD mode execution
 * - Modified or rejected
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

// Plan step status
export enum StepStatus {
    Pending = 'pending',
    InProgress = 'in_progress',
    Completed = 'completed',
    Skipped = 'skipped',
    Failed = 'failed',
}

// Plan step type
export enum StepType {
    Read = 'read',
    Search = 'search',
    Edit = 'edit',
    Create = 'create',
    Delete = 'delete',
    Execute = 'execute',
    Verify = 'verify',
}

// A single step in a plan
export interface PlanStep {
    id: string;
    description: string;
    type: StepType;
    status: StepStatus;
    files?: string[];
    dependencies?: string[];  // IDs of steps this depends on
    output?: string;
    error?: string;
}

// An execution plan
export interface ExecutionPlan {
    id: string;
    goal: string;
    createdAt: number;
    approvedAt?: number;
    status: 'draft' | 'approved' | 'executing' | 'completed' | 'failed';
    steps: PlanStep[];
    metadata?: Record<string, unknown>;
}

// Tree item types
type PlanTreeItem = PlanItem | PlanStepItem | PlanInfoItem;

/**
 * Tree item representing an execution plan
 */
class PlanItem extends vscode.TreeItem {
    constructor(
        public readonly plan: ExecutionPlan,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(plan.goal.slice(0, 50), collapsibleState);

        this.id = plan.id;
        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = `plan-${plan.status}`;
    }

    private _getDescription(): string {
        const completed = this.plan.steps.filter(s => s.status === StepStatus.Completed).length;
        const total = this.plan.steps.length;
        return `${this._getStatusLabel()} • ${completed}/${total} steps`;
    }

    private _getStatusLabel(): string {
        switch (this.plan.status) {
            case 'draft': return '📝 Draft';
            case 'approved': return '✅ Approved';
            case 'executing': return '🔄 Executing';
            case 'completed': return '✅ Done';
            case 'failed': return '❌ Failed';
            default: return this.plan.status;
        }
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### ${this.plan.goal}\n\n`);
        md.appendMarkdown(`**Status:** ${this._getStatusLabel()}\n\n`);
        md.appendMarkdown(`**Created:** ${new Date(this.plan.createdAt).toLocaleString()}\n\n`);
        md.appendMarkdown(`**Steps:** ${this.plan.steps.length}\n\n`);

        if (this.plan.approvedAt) {
            md.appendMarkdown(`**Approved:** ${new Date(this.plan.approvedAt).toLocaleString()}\n\n`);
        }

        return md;
    }

    private _getIcon(): vscode.ThemeIcon {
        switch (this.plan.status) {
            case 'draft':
                return new vscode.ThemeIcon('notebook', new vscode.ThemeColor('charts.yellow'));
            case 'approved':
                return new vscode.ThemeIcon('notebook', new vscode.ThemeColor('charts.blue'));
            case 'executing':
                return new vscode.ThemeIcon('sync~spin', new vscode.ThemeColor('charts.blue'));
            case 'completed':
                return new vscode.ThemeIcon('notebook', new vscode.ThemeColor('charts.green'));
            case 'failed':
                return new vscode.ThemeIcon('notebook', new vscode.ThemeColor('charts.red'));
            default:
                return new vscode.ThemeIcon('notebook');
        }
    }
}

/**
 * Tree item representing a plan step
 */
class PlanStepItem extends vscode.TreeItem {
    constructor(
        public readonly step: PlanStep,
        public readonly planId: string,
        public readonly index: number
    ) {
        super(`${index + 1}. ${step.description.slice(0, 60)}`, vscode.TreeItemCollapsibleState.None);

        this.id = `${planId}-${step.id}`;
        this.description = this._getDescription();
        this.tooltip = this._getTooltip();
        this.iconPath = this._getIcon();
        this.contextValue = `step-${step.status}`;
    }

    private _getDescription(): string {
        const files = this.step.files?.length ? ` (${this.step.files.length} files)` : '';
        return `${this._getStatusIcon()}${files}`;
    }

    private _getStatusIcon(): string {
        switch (this.step.status) {
            case StepStatus.Pending: return '⏳';
            case StepStatus.InProgress: return '🔄';
            case StepStatus.Completed: return '✅';
            case StepStatus.Skipped: return '⏭️';
            case StepStatus.Failed: return '❌';
            default: return '•';
        }
    }

    private _getTooltip(): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`**${this.step.description}**\n\n`);
        md.appendMarkdown(`**Type:** ${this.step.type}\n\n`);
        md.appendMarkdown(`**Status:** ${this.step.status}\n\n`);

        if (this.step.files?.length) {
            md.appendMarkdown(`**Files:**\n`);
            this.step.files.forEach(f => md.appendMarkdown(`- ${f}\n`));
        }

        if (this.step.output) {
            md.appendMarkdown(`\n**Output:**\n\`\`\`\n${this.step.output.slice(0, 200)}\n\`\`\``);
        }

        if (this.step.error) {
            md.appendMarkdown(`\n**Error:** ${this.step.error}`);
        }

        return md;
    }

    private _getIcon(): vscode.ThemeIcon {
        // Icon based on step type
        switch (this.step.type) {
            case StepType.Read:
                return new vscode.ThemeIcon('file-text');
            case StepType.Search:
                return new vscode.ThemeIcon('search');
            case StepType.Edit:
                return new vscode.ThemeIcon('edit');
            case StepType.Create:
                return new vscode.ThemeIcon('new-file');
            case StepType.Delete:
                return new vscode.ThemeIcon('trash');
            case StepType.Execute:
                return new vscode.ThemeIcon('terminal');
            case StepType.Verify:
                return new vscode.ThemeIcon('check-all');
            default:
                return new vscode.ThemeIcon('circle-outline');
        }
    }
}

/**
 * Tree item for plan info/metadata
 */
class PlanInfoItem extends vscode.TreeItem {
    constructor(label: string, value: string, icon?: string) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.description = value;
        this.iconPath = icon ? new vscode.ThemeIcon(icon) : undefined;
        this.contextValue = 'plan-info';
    }
}

/**
 * Plans View TreeDataProvider
 */
export class PlansViewProvider implements vscode.TreeDataProvider<PlanTreeItem>, vscode.Disposable {
    private _onDidChangeTreeData: vscode.EventEmitter<PlanTreeItem | undefined | null | void> =
        new vscode.EventEmitter<PlanTreeItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<PlanTreeItem | undefined | null | void> =
        this._onDidChangeTreeData.event;

    private _plans: Map<string, ExecutionPlan> = new Map();
    private _disposables: vscode.Disposable[] = [];

    constructor(
        private readonly _client: VictorClient,
        private readonly _outputChannel?: vscode.OutputChannel
    ) {
        // Load saved plans on init
        this._loadPlans();
    }

    dispose(): void {
        this._disposables.forEach(d => d.dispose());
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    // =========================================================================
    // TreeDataProvider Implementation
    // =========================================================================

    getTreeItem(element: PlanTreeItem): vscode.TreeItem {
        return element;
    }

    async getChildren(element?: PlanTreeItem): Promise<PlanTreeItem[]> {
        if (!element) {
            // Root level: show all plans
            return this._getPlanItems();
        }

        if (element instanceof PlanItem) {
            // Show steps for this plan
            return this._getStepItems(element.plan);
        }

        return [];
    }

    getParent(element: PlanTreeItem): vscode.ProviderResult<PlanTreeItem> {
        if (element instanceof PlanStepItem) {
            const plan = this._plans.get(element.planId);
            if (plan) {
                return new PlanItem(plan, vscode.TreeItemCollapsibleState.Expanded);
            }
        }
        return null;
    }

    // =========================================================================
    // Plan Management
    // =========================================================================

    /**
     * Add or update a plan
     */
    addPlan(plan: ExecutionPlan): void {
        this._plans.set(plan.id, plan);
        this.refresh();
        this._log(`Plan added: ${plan.id} - ${plan.goal}`);
    }

    /**
     * Update an existing plan
     */
    updatePlan(id: string, updates: Partial<ExecutionPlan>): void {
        const plan = this._plans.get(id);
        if (plan) {
            Object.assign(plan, updates);
            this.refresh();
        }
    }

    /**
     * Update a step in a plan
     */
    updateStep(planId: string, stepId: string, updates: Partial<PlanStep>): void {
        const plan = this._plans.get(planId);
        if (plan) {
            const step = plan.steps.find(s => s.id === stepId);
            if (step) {
                Object.assign(step, updates);
                this.refresh();
            }
        }
    }

    /**
     * Approve a plan for execution
     */
    async approvePlan(id: string): Promise<void> {
        const plan = this._plans.get(id);
        if (plan && plan.status === 'draft') {
            try {
                // Send approval to backend
                const result = await this._client.approvePlan(id);
                if (result.success) {
                    plan.status = 'approved';
                    plan.approvedAt = Date.now();
                    this.refresh();
                    this._log(`Plan approved: ${id}`);
                } else {
                    vscode.window.showErrorMessage(`Failed to approve plan: ${result.message}`);
                }
            } catch (error) {
                // Update locally even if backend fails
                plan.status = 'approved';
                plan.approvedAt = Date.now();
                this.refresh();
                this._log(`Plan approved locally (backend unavailable): ${id}`);
            }
        }
    }

    /**
     * Execute an approved plan
     */
    async executePlan(id: string): Promise<void> {
        const plan = this._plans.get(id);
        if (plan && plan.status === 'approved') {
            try {
                // Send execute request to backend
                const result = await this._client.executePlan(id);
                if (result.success) {
                    plan.status = 'executing';
                    this.refresh();
                    this._log(`Plan execution started: ${id}`);
                } else {
                    vscode.window.showErrorMessage(`Failed to execute plan: ${result.message}`);
                }
            } catch (error) {
                // Update locally even if backend fails
                plan.status = 'executing';
                this.refresh();
                this._log(`Plan execution started locally (backend unavailable): ${id}`);
            }
        }
    }

    /**
     * Remove a plan
     */
    removePlan(id: string): void {
        this._plans.delete(id);
        this.refresh();
    }

    /**
     * Clear all completed/failed plans
     */
    clearCompleted(): void {
        for (const [id, plan] of this._plans.entries()) {
            if (plan.status === 'completed' || plan.status === 'failed') {
                this._plans.delete(id);
            }
        }
        this.refresh();
    }

    /**
     * Get current plan (most recent executing or approved)
     */
    getCurrentPlan(): ExecutionPlan | undefined {
        for (const plan of this._plans.values()) {
            if (plan.status === 'executing' || plan.status === 'approved') {
                return plan;
            }
        }
        return undefined;
    }

    // =========================================================================
    // Private Methods
    // =========================================================================

    private _getPlanItems(): PlanTreeItem[] {
        if (this._plans.size === 0) {
            return [new PlanInfoItem('No plans', 'Use PLAN mode to create a plan', 'info')];
        }

        const items: PlanItem[] = [];

        // Sort: executing first, then approved, then by creation time
        const sorted = Array.from(this._plans.values()).sort((a, b) => {
            const statusOrder = {
                'executing': 0,
                'approved': 1,
                'draft': 2,
                'completed': 3,
                'failed': 4,
            };
            const statusDiff = statusOrder[a.status] - statusOrder[b.status];
            if (statusDiff !== 0) {return statusDiff;}
            return b.createdAt - a.createdAt;  // Newer first
        });

        for (const plan of sorted) {
            items.push(new PlanItem(
                plan,
                vscode.TreeItemCollapsibleState.Collapsed
            ));
        }

        return items;
    }

    private _getStepItems(plan: ExecutionPlan): PlanTreeItem[] {
        return plan.steps.map((step, index) => new PlanStepItem(step, plan.id, index));
    }

    private async _loadPlans(): Promise<void> {
        try {
            // Load plans from backend
            const backendPlans = await this._client.listPlans();

            for (const p of backendPlans) {
                // Convert backend plan format to local format
                const steps: PlanStep[] = p.steps.map((s, i) => ({
                    id: `step-${i}`,
                    description: typeof s === 'string' ? s : s.description,
                    type: StepType.Execute,
                    status: s.status === 'completed' ? StepStatus.Completed :
                           s.status === 'in_progress' ? StepStatus.InProgress :
                           StepStatus.Pending,
                }));

                const plan: ExecutionPlan = {
                    id: p.id,
                    goal: p.title || p.description || 'Untitled Plan',
                    createdAt: p.created_at * 1000, // Convert to milliseconds
                    approvedAt: p.approved_at ? p.approved_at * 1000 : undefined,
                    status: p.status as ExecutionPlan['status'],
                    steps,
                };

                this._plans.set(p.id, plan);
            }

            if (backendPlans.length > 0) {
                this._log(`Loaded ${backendPlans.length} plans from backend`);
                this.refresh();
            }
        } catch (error) {
            // Backend unavailable - start with empty plans
            this._log('Could not load plans from backend, starting with empty list');
        }
    }

    private _log(message: string): void {
        this._outputChannel?.appendLine(`[Plans] ${message}`);
    }
}

/**
 * Register plan-related commands
 */
export function registerPlanCommands(
    context: vscode.ExtensionContext,
    plansView: PlansViewProvider
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshPlans', () => {
            plansView.refresh();
        }),
        vscode.commands.registerCommand('victor.clearPlans', () => {
            plansView.clearCompleted();
            vscode.window.showInformationMessage('Cleared completed plans');
        }),
        vscode.commands.registerCommand('victor.approvePlan', async (item: PlanItem) => {
            if (item?.plan?.id) {
                await plansView.approvePlan(item.plan.id);
                vscode.window.showInformationMessage(`Approved plan: ${item.plan.goal.slice(0, 50)}`);
            }
        }),
        vscode.commands.registerCommand('victor.executePlan', async (item: PlanItem) => {
            if (item?.plan?.id) {
                await plansView.executePlan(item.plan.id);
                vscode.window.showInformationMessage(`Executing plan: ${item.plan.goal.slice(0, 50)}`);
            }
        }),
        vscode.commands.registerCommand('victor.deletePlan', async (item: PlanItem) => {
            if (item?.plan?.id) {
                const confirm = await vscode.window.showWarningMessage(
                    `Delete plan: "${item.plan.goal.slice(0, 50)}"?`,
                    { modal: true },
                    'Delete',
                    'Cancel'
                );
                if (confirm === 'Delete') {
                    plansView.removePlan(item.plan.id);
                    vscode.window.showInformationMessage('Plan deleted');
                }
            }
        })
    );
}

/**
 * Create demo plan for testing
 */
export function createDemoPlan(): ExecutionPlan {
    return {
        id: `plan-${Date.now()}`,
        goal: 'Refactor authentication module to use JWT tokens',
        createdAt: Date.now() - 60000,
        status: 'draft',
        steps: [
            {
                id: 'step-1',
                description: 'Read current auth implementation',
                type: StepType.Read,
                status: StepStatus.Completed,
                files: ['src/auth/handler.py', 'src/auth/models.py'],
            },
            {
                id: 'step-2',
                description: 'Search for all auth usages',
                type: StepType.Search,
                status: StepStatus.Completed,
            },
            {
                id: 'step-3',
                description: 'Create JWT utility module',
                type: StepType.Create,
                status: StepStatus.InProgress,
                files: ['src/auth/jwt_utils.py'],
            },
            {
                id: 'step-4',
                description: 'Update auth handler to use JWT',
                type: StepType.Edit,
                status: StepStatus.Pending,
                files: ['src/auth/handler.py'],
                dependencies: ['step-3'],
            },
            {
                id: 'step-5',
                description: 'Run tests to verify changes',
                type: StepType.Verify,
                status: StepStatus.Pending,
                dependencies: ['step-4'],
            },
        ],
    };
}
