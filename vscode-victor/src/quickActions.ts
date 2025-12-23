/**
 * Quick Actions Provider (Cmd+K)
 *
 * Provides a universal quick action menu for common AI operations.
 * Context-aware suggestions based on selection and file type.
 */

import * as vscode from 'vscode';
import { VictorClient } from './victorClient';

export interface QuickAction {
    label: string;
    description: string;
    icon: string;
    action: string;
    shortcut?: string;
    category: 'edit' | 'explain' | 'generate' | 'refactor' | 'debug';
}

/**
 * Built-in quick actions
 */
const QUICK_ACTIONS: QuickAction[] = [
    // Edit actions
    {
        label: 'Fix',
        description: 'Fix bugs and issues in selected code',
        icon: '$(bug)',
        action: 'fix',
        shortcut: 'f',
        category: 'edit',
    },
    {
        label: 'Refactor',
        description: 'Refactor and improve code structure',
        icon: '$(edit)',
        action: 'refactor',
        shortcut: 'r',
        category: 'refactor',
    },
    {
        label: 'Optimize',
        description: 'Optimize code for performance',
        icon: '$(zap)',
        action: 'optimize',
        shortcut: 'o',
        category: 'edit',
    },
    {
        label: 'Simplify',
        description: 'Simplify and clean up code',
        icon: '$(sparkle)',
        action: 'simplify',
        category: 'refactor',
    },

    // Explain actions
    {
        label: 'Explain',
        description: 'Explain what this code does',
        icon: '$(question)',
        action: 'explain',
        shortcut: 'e',
        category: 'explain',
    },
    {
        label: 'Explain Error',
        description: 'Explain the error and how to fix it',
        icon: '$(error)',
        action: 'explain_error',
        category: 'explain',
    },

    // Generate actions
    {
        label: 'Generate Tests',
        description: 'Generate unit tests for this code',
        icon: '$(beaker)',
        action: 'test',
        shortcut: 't',
        category: 'generate',
    },
    {
        label: 'Add Documentation',
        description: 'Add documentation and comments',
        icon: '$(book)',
        action: 'document',
        shortcut: 'd',
        category: 'generate',
    },
    {
        label: 'Add Types',
        description: 'Add type annotations',
        icon: '$(symbol-type-parameter)',
        action: 'add_types',
        category: 'generate',
    },
    {
        label: 'Add Error Handling',
        description: 'Add try-catch and error handling',
        icon: '$(shield)',
        action: 'error_handling',
        category: 'generate',
    },

    // Debug actions
    {
        label: 'Add Logging',
        description: 'Add debug logging statements',
        icon: '$(output)',
        action: 'add_logging',
        category: 'debug',
    },
    {
        label: 'Find Issues',
        description: 'Review code and find potential issues',
        icon: '$(search)',
        action: 'review',
        category: 'debug',
    },
];

/**
 * Quick pick item with action data
 */
interface ActionQuickPickItem extends vscode.QuickPickItem {
    action: QuickAction;
}

export class QuickActionsProvider implements vscode.Disposable {
    private _disposables: vscode.Disposable[] = [];
    private _recentActions: string[] = [];
    private _maxRecentActions = 5;

    constructor(
        private readonly _client: VictorClient,
        private readonly _onAction: (action: string, text: string, languageId: string) => Promise<void>,
        private readonly _log?: vscode.OutputChannel
    ) {
        // Register Cmd+K command
        this._disposables.push(
            vscode.commands.registerCommand('victor.quickAction', () => this.showQuickActions()),
            vscode.commands.registerCommand('victor.quickActionWithContext', (context: string) =>
                this.showQuickActions(context)
            ),
        );

        // Register individual action commands
        for (const action of QUICK_ACTIONS) {
            this._disposables.push(
                vscode.commands.registerCommand(`victor.quick.${action.action}`, () =>
                    this._executeAction(action)
                )
            );
        }

        // Load recent actions from storage
        this._loadRecentActions();
    }

    public dispose(): void {
        this._disposables.forEach(d => d.dispose());
    }

    /**
     * Show the quick actions menu
     */
    public async showQuickActions(prefilter?: string): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        const selection = editor.selection;
        const hasSelection = !selection.isEmpty;
        const languageId = editor.document.languageId;

        // Get context-aware actions
        const contextualActions = this._getContextualActions(languageId, hasSelection);

        // Build quick pick items
        const items = this._buildQuickPickItems(contextualActions, prefilter);

        // Show quick pick with search
        const quickPick = vscode.window.createQuickPick<ActionQuickPickItem>();
        quickPick.items = items;
        quickPick.placeholder = hasSelection
            ? 'What would you like to do with the selected code?'
            : 'What would you like to do? (select code for more options)';
        quickPick.matchOnDescription = true;
        quickPick.matchOnDetail = true;

        // Handle selection
        quickPick.onDidAccept(async () => {
            const selected = quickPick.selectedItems[0];
            if (selected) {
                quickPick.hide();
                await this._executeAction(selected.action);
            }
        });

        // Handle keyboard shortcuts
        quickPick.onDidChangeValue((value) => {
            // Check if value matches a shortcut
            const action = QUICK_ACTIONS.find(a => a.shortcut === value.toLowerCase());
            if (action) {
                quickPick.hide();
                this._executeAction(action);
            }
        });

        quickPick.onDidHide(() => quickPick.dispose());
        quickPick.show();
    }

    /**
     * Get context-aware actions based on language and selection
     */
    private _getContextualActions(languageId: string, hasSelection: boolean): QuickAction[] {
        let actions = [...QUICK_ACTIONS];

        // Add language-specific actions
        if (['typescript', 'javascript', 'typescriptreact', 'javascriptreact'].includes(languageId)) {
            actions.push({
                label: 'Convert to Async',
                description: 'Convert to async/await pattern',
                icon: '$(sync)',
                action: 'convert_async',
                category: 'refactor',
            });
        }

        if (['python'].includes(languageId)) {
            actions.push({
                label: 'Add Type Hints',
                description: 'Add Python type hints',
                icon: '$(symbol-type-parameter)',
                action: 'add_type_hints',
                category: 'generate',
            });
        }

        if (['typescript', 'javascript', 'python', 'java', 'csharp'].includes(languageId)) {
            actions.push({
                label: 'Extract Function',
                description: 'Extract selection into a function',
                icon: '$(symbol-function)',
                action: 'extract_function',
                category: 'refactor',
            });
        }

        // Sort by recent usage
        actions = this._sortByRecent(actions);

        return actions;
    }

    /**
     * Build quick pick items with categories
     */
    private _buildQuickPickItems(
        actions: QuickAction[],
        prefilter?: string
    ): ActionQuickPickItem[] {
        const items: ActionQuickPickItem[] = [];

        // Add recent actions first if no prefilter
        if (!prefilter && this._recentActions.length > 0) {
            items.push({
                label: 'Recent',
                kind: vscode.QuickPickItemKind.Separator,
                action: {} as QuickAction,
            });

            for (const recentAction of this._recentActions) {
                const action = actions.find(a => a.action === recentAction);
                if (action) {
                    items.push(this._actionToQuickPickItem(action, true));
                }
            }
        }

        // Group by category
        const categories: { [key: string]: QuickAction[] } = {
            edit: [],
            explain: [],
            generate: [],
            refactor: [],
            debug: [],
        };

        for (const action of actions) {
            if (!prefilter || action.label.toLowerCase().includes(prefilter.toLowerCase())) {
                categories[action.category].push(action);
            }
        }

        // Add categorized items
        const categoryLabels: { [key: string]: string } = {
            edit: 'Edit',
            explain: 'Understand',
            generate: 'Generate',
            refactor: 'Refactor',
            debug: 'Debug',
        };

        for (const [category, categoryActions] of Object.entries(categories)) {
            if (categoryActions.length > 0) {
                items.push({
                    label: categoryLabels[category],
                    kind: vscode.QuickPickItemKind.Separator,
                    action: {} as QuickAction,
                });

                for (const action of categoryActions) {
                    // Skip if already in recent
                    if (!this._recentActions.includes(action.action)) {
                        items.push(this._actionToQuickPickItem(action));
                    }
                }
            }
        }

        return items;
    }

    /**
     * Convert action to quick pick item
     */
    private _actionToQuickPickItem(action: QuickAction, isRecent = false): ActionQuickPickItem {
        return {
            label: `${action.icon} ${action.label}`,
            description: action.shortcut ? `(${action.shortcut})` : undefined,
            detail: action.description,
            action,
        };
    }

    /**
     * Sort actions by recent usage
     */
    private _sortByRecent(actions: QuickAction[]): QuickAction[] {
        return [...actions].sort((a, b) => {
            const aIndex = this._recentActions.indexOf(a.action);
            const bIndex = this._recentActions.indexOf(b.action);

            if (aIndex >= 0 && bIndex >= 0) {
                return aIndex - bIndex;
            }
            if (aIndex >= 0) return -1;
            if (bIndex >= 0) return 1;
            return 0;
        });
    }

    /**
     * Execute an action
     */
    private async _executeAction(action: QuickAction): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const selection = editor.selection;
        const text = selection.isEmpty
            ? editor.document.getText()
            : editor.document.getText(selection);

        // Track recent action
        this._addRecentAction(action.action);

        this._log?.appendLine(`[QuickActions] Executing: ${action.action}`);

        try {
            await this._onAction(action.action, text, editor.document.languageId);
        } catch (error) {
            this._log?.appendLine(`[QuickActions] Error: ${error}`);
            vscode.window.showErrorMessage(`Action failed: ${error}`);
        }
    }

    /**
     * Add action to recent list
     */
    private _addRecentAction(action: string): void {
        // Remove if already exists
        const index = this._recentActions.indexOf(action);
        if (index >= 0) {
            this._recentActions.splice(index, 1);
        }

        // Add to front
        this._recentActions.unshift(action);

        // Trim to max size
        if (this._recentActions.length > this._maxRecentActions) {
            this._recentActions = this._recentActions.slice(0, this._maxRecentActions);
        }

        // Persist
        this._saveRecentActions();
    }

    /**
     * Load recent actions from storage
     */
    private _loadRecentActions(): void {
        try {
            const stored = vscode.workspace.getConfiguration('victor').get<string[]>('recentQuickActions');
            if (stored) {
                this._recentActions = stored;
            }
        } catch {
            // Ignore errors
        }
    }

    /**
     * Save recent actions to storage
     */
    private _saveRecentActions(): void {
        try {
            vscode.workspace.getConfiguration('victor').update(
                'recentQuickActions',
                this._recentActions,
                vscode.ConfigurationTarget.Global
            );
        } catch {
            // Ignore errors
        }
    }
}
