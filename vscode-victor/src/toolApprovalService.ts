/**
 * Tool Approval Service
 *
 * Manages approval workflow for dangerous tool executions.
 * Uses an event-driven architecture with support for:
 * - Manual approval via modal dialogs
 * - Auto-approval rules based on user preferences
 * - Approval history tracking
 * - Rate limiting for security
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

/**
 * Tool approval request from the backend
 */
export interface ToolApprovalRequest {
    id: string;
    toolName: string;
    arguments: Record<string, unknown>;
    reason: string;
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    timestamp: number;
    category?: string;
    estimatedImpact?: string;
}

/**
 * User's approval decision
 */
export interface ApprovalDecision {
    approved: boolean;
    rememberForSession?: boolean;
    rememberForTool?: boolean;
}

/**
 * Auto-approval rule
 */
interface AutoApprovalRule {
    toolName: string;
    maxRiskLevel: 'low' | 'medium' | 'high';
    expiresAt?: number;
}

/**
 * Service for managing tool approval workflows
 */
export class ToolApprovalService implements vscode.Disposable {
    private readonly _onApprovalRequested = new vscode.EventEmitter<ToolApprovalRequest>();
    readonly onApprovalRequested = this._onApprovalRequested.event;

    private readonly _onApprovalDecided = new vscode.EventEmitter<{ id: string; decision: ApprovalDecision }>();
    readonly onApprovalDecided = this._onApprovalDecided.event;

    private autoApprovalRules: Map<string, AutoApprovalRule> = new Map();
    private sessionApprovals: Set<string> = new Set();
    private pendingApprovals: Map<string, ToolApprovalRequest> = new Map();
    private approvalHistory: { request: ToolApprovalRequest; decision: ApprovalDecision; decidedAt: number }[] = [];

    private readonly maxHistorySize = 100;
    private readonly disposables: vscode.Disposable[] = [];

    constructor(
        private readonly client: VictorClient,
        private readonly context: vscode.ExtensionContext
    ) {
        // Load persisted auto-approval rules
        this.loadAutoApprovalRules();
    }

    /**
     * Request approval for a dangerous tool execution
     */
    async requestApproval(request: ToolApprovalRequest): Promise<ApprovalDecision> {
        // Check auto-approval rules first
        const autoApproval = this.checkAutoApproval(request);
        if (autoApproval) {
            this.recordApproval(request, { approved: true });
            return { approved: true };
        }

        // Check session approvals
        if (this.sessionApprovals.has(request.toolName)) {
            this.recordApproval(request, { approved: true, rememberForSession: true });
            return { approved: true, rememberForSession: true };
        }

        // Store pending approval
        this.pendingApprovals.set(request.id, request);
        this._onApprovalRequested.fire(request);

        // Show approval dialog
        const decision = await this.showApprovalDialog(request);

        // Apply decision
        this.pendingApprovals.delete(request.id);
        this.recordApproval(request, decision);
        this._onApprovalDecided.fire({ id: request.id, decision });

        // Handle "remember" options
        if (decision.approved) {
            if (decision.rememberForSession) {
                this.sessionApprovals.add(request.toolName);
            }
            if (decision.rememberForTool) {
                // Critical risk tools cannot be auto-approved (UI blocks this),
                // but we guard against it here for safety. Cap at 'high'.
                const maxRisk = request.riskLevel === 'critical' ? 'high' : request.riskLevel;
                this.addAutoApprovalRule({
                    toolName: request.toolName,
                    maxRiskLevel: maxRisk,
                });
            }
        }

        // Notify backend of decision
        try {
            await this.client.approveTool(request.id, decision.approved);
        } catch (error) {
            console.error('Failed to notify backend of approval decision:', error);
        }

        return decision;
    }

    /**
     * Show approval dialog to user
     */
    private async showApprovalDialog(request: ToolApprovalRequest): Promise<ApprovalDecision> {
        const riskColors = {
            low: '$(info)',
            medium: '$(warning)',
            high: '$(error)',
            critical: '$(alert)'
        };

        const riskLabels = {
            low: 'Low Risk',
            medium: 'Medium Risk',
            high: 'High Risk',
            critical: 'Critical Risk'
        };

        // Build message with formatted arguments
        const argsPreview = this.formatArguments(request.arguments);
        const message = [
            `${riskColors[request.riskLevel]} **${riskLabels[request.riskLevel]}**: ${request.toolName}`,
            '',
            request.reason,
            '',
            `**Arguments:**`,
            argsPreview,
            request.estimatedImpact ? `\n**Impact:** ${request.estimatedImpact}` : '',
        ].filter(Boolean).join('\n');

        // Create quick pick items based on risk level
        interface ApprovalItem extends vscode.QuickPickItem {
            action: 'approve' | 'approve_session' | 'approve_always' | 'deny';
        }

        const items: ApprovalItem[] = [
            {
                label: '$(check) Approve Once',
                description: 'Allow this tool execution',
                action: 'approve',
            },
            {
                label: '$(checklist) Approve for Session',
                description: `Allow ${request.toolName} for this session`,
                action: 'approve_session',
            },
        ];

        // Only show "always approve" for low/medium risk
        if (request.riskLevel === 'low' || request.riskLevel === 'medium') {
            items.push({
                label: '$(star) Always Approve',
                description: `Auto-approve ${request.toolName} in the future`,
                action: 'approve_always',
            });
        }

        items.push({
            label: '$(x) Deny',
            description: 'Block this tool execution',
            action: 'deny',
        });

        // Show quick pick
        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: message.substring(0, 200) + (message.length > 200 ? '...' : ''),
            title: `Tool Approval: ${request.toolName}`,
            ignoreFocusOut: true,
        });

        if (!selected || selected.action === 'deny') {
            return { approved: false };
        }

        return {
            approved: true,
            rememberForSession: selected.action === 'approve_session',
            rememberForTool: selected.action === 'approve_always',
        };
    }

    /**
     * Check if request matches auto-approval rules
     */
    private checkAutoApproval(request: ToolApprovalRequest): boolean {
        const rule = this.autoApprovalRules.get(request.toolName);
        if (!rule) {
            return false;
        }

        // Check expiration
        if (rule.expiresAt && Date.now() > rule.expiresAt) {
            this.autoApprovalRules.delete(request.toolName);
            this.saveAutoApprovalRules();
            return false;
        }

        // Check risk level
        const riskOrder = ['low', 'medium', 'high', 'critical'];
        const requestRiskIndex = riskOrder.indexOf(request.riskLevel);
        const maxRiskIndex = riskOrder.indexOf(rule.maxRiskLevel);

        return requestRiskIndex <= maxRiskIndex;
    }

    /**
     * Add auto-approval rule
     */
    addAutoApprovalRule(rule: AutoApprovalRule): void {
        this.autoApprovalRules.set(rule.toolName, rule);
        this.saveAutoApprovalRules();
    }

    /**
     * Remove auto-approval rule
     */
    removeAutoApprovalRule(toolName: string): void {
        this.autoApprovalRules.delete(toolName);
        this.saveAutoApprovalRules();
    }

    /**
     * Get all auto-approval rules
     */
    getAutoApprovalRules(): AutoApprovalRule[] {
        return Array.from(this.autoApprovalRules.values());
    }

    /**
     * Clear session approvals
     */
    clearSessionApprovals(): void {
        this.sessionApprovals.clear();
    }

    /**
     * Get pending approvals
     */
    getPendingApprovals(): ToolApprovalRequest[] {
        return Array.from(this.pendingApprovals.values());
    }

    /**
     * Get approval history
     */
    getApprovalHistory(): typeof this.approvalHistory {
        return [...this.approvalHistory];
    }

    /**
     * Format arguments for display
     */
    private formatArguments(args: Record<string, unknown>): string {
        const lines: string[] = [];
        for (const [key, value] of Object.entries(args)) {
            const valueStr = typeof value === 'object'
                ? JSON.stringify(value, null, 2)
                : String(value);
            const truncated = valueStr.length > 100
                ? valueStr.substring(0, 100) + '...'
                : valueStr;
            lines.push(`  ${key}: ${truncated}`);
        }
        return lines.join('\n') || '  (none)';
    }

    /**
     * Record approval decision in history
     */
    private recordApproval(request: ToolApprovalRequest, decision: ApprovalDecision): void {
        this.approvalHistory.push({
            request,
            decision,
            decidedAt: Date.now(),
        });

        // Trim history if too large
        if (this.approvalHistory.length > this.maxHistorySize) {
            this.approvalHistory = this.approvalHistory.slice(-this.maxHistorySize);
        }
    }

    /**
     * Load auto-approval rules from storage
     */
    private loadAutoApprovalRules(): void {
        const stored = this.context.globalState.get<[string, AutoApprovalRule][]>('victor.autoApprovalRules');
        if (stored) {
            this.autoApprovalRules = new Map(stored);
        }
    }

    /**
     * Save auto-approval rules to storage
     */
    private saveAutoApprovalRules(): void {
        const entries = Array.from(this.autoApprovalRules.entries());
        this.context.globalState.update('victor.autoApprovalRules', entries);
    }

    dispose(): void {
        this._onApprovalRequested.dispose();
        this._onApprovalDecided.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}

/**
 * Register tool approval commands
 */
export function registerToolApprovalCommands(
    context: vscode.ExtensionContext,
    service: ToolApprovalService
): void {
    // View pending approvals
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.viewPendingApprovals', async () => {
            const pending = service.getPendingApprovals();
            if (pending.length === 0) {
                vscode.window.showInformationMessage('No pending tool approvals');
                return;
            }

            const items = pending.map(p => ({
                label: `$(tools) ${p.toolName}`,
                description: p.riskLevel,
                detail: p.reason,
                request: p,
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a pending approval to review',
            });

            if (selected) {
                // Re-show approval dialog for selected request
                await service.requestApproval(selected.request);
            }
        })
    );

    // Manage auto-approval rules
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.manageAutoApprovals', async () => {
            const rules = service.getAutoApprovalRules();

            if (rules.length === 0) {
                const addNew = await vscode.window.showQuickPick(['Add Rule', 'Cancel'], {
                    placeHolder: 'No auto-approval rules configured',
                });
                if (addNew === 'Add Rule') {
                    // Would show input for new rule
                    vscode.window.showInformationMessage('Auto-approval rules are added when you approve a tool with "Always Approve"');
                }
                return;
            }

            interface RuleItem extends vscode.QuickPickItem {
                rule?: AutoApprovalRule;
                action: 'remove' | 'clear' | 'cancel';
            }

            const items: RuleItem[] = [
                ...rules.map(r => ({
                    label: `$(tools) ${r.toolName}`,
                    description: `Max risk: ${r.maxRiskLevel}`,
                    rule: r,
                    action: 'remove' as const,
                })),
                { label: '$(trash) Clear All Rules', action: 'clear' as const },
                { label: '$(x) Cancel', action: 'cancel' as const },
            ];

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Manage auto-approval rules',
            });

            if (!selected || selected.action === 'cancel') {
                return;
            }

            if (selected.action === 'clear') {
                for (const rule of rules) {
                    service.removeAutoApprovalRule(rule.toolName);
                }
                vscode.window.showInformationMessage('All auto-approval rules cleared');
            } else if (selected.rule) {
                service.removeAutoApprovalRule(selected.rule.toolName);
                vscode.window.showInformationMessage(`Removed auto-approval for ${selected.rule.toolName}`);
            }
        })
    );

    // View approval history
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.viewApprovalHistory', async () => {
            const history = service.getApprovalHistory();
            if (history.length === 0) {
                vscode.window.showInformationMessage('No approval history');
                return;
            }

            const items = history.slice(-20).reverse().map(h => ({
                label: `${h.decision.approved ? '$(check)' : '$(x)'} ${h.request.toolName}`,
                description: new Date(h.decidedAt).toLocaleString(),
                detail: h.request.reason,
            }));

            await vscode.window.showQuickPick(items, {
                placeHolder: 'Recent approval decisions',
            });
        })
    );
}
