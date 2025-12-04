/**
 * Tools View Provider
 *
 * Provides a tree view of available Victor tools organized by category.
 */

import * as vscode from 'vscode';

interface ToolInfo {
    name: string;
    description: string;
    category: string;
    costTier: 'free' | 'low' | 'medium' | 'high';
}

// Tool categories and their tools
const TOOL_CATEGORIES: Record<string, ToolInfo[]> = {
    'File Operations': [
        { name: 'read_file', description: 'Read file contents', category: 'File Operations', costTier: 'free' },
        { name: 'write_file', description: 'Write or create files', category: 'File Operations', costTier: 'free' },
        { name: 'edit_files', description: 'Edit multiple files', category: 'File Operations', costTier: 'free' },
        { name: 'list_directory', description: 'List directory contents', category: 'File Operations', costTier: 'free' },
        { name: 'glob', description: 'Find files by pattern', category: 'File Operations', costTier: 'free' },
    ],
    'Code Search': [
        { name: 'code_search', description: 'Search code by pattern', category: 'Code Search', costTier: 'free' },
        { name: 'semantic_code_search', description: 'Search by meaning', category: 'Code Search', costTier: 'low' },
        { name: 'grep', description: 'Search with regex', category: 'Code Search', costTier: 'free' },
    ],
    'Git': [
        { name: 'git_status', description: 'Show git status', category: 'Git', costTier: 'free' },
        { name: 'git_diff', description: 'Show git diff', category: 'Git', costTier: 'free' },
        { name: 'git_log', description: 'Show commit history', category: 'Git', costTier: 'free' },
        { name: 'git_commit', description: 'Create a commit', category: 'Git', costTier: 'free' },
        { name: 'git_push', description: 'Push to remote', category: 'Git', costTier: 'free' },
    ],
    'Shell': [
        { name: 'bash', description: 'Run shell commands', category: 'Shell', costTier: 'free' },
    ],
    'Analysis': [
        { name: 'code_review', description: 'Review code quality', category: 'Analysis', costTier: 'low' },
        { name: 'dependency_graph', description: 'Analyze dependencies', category: 'Analysis', costTier: 'low' },
        { name: 'plan_files', description: 'Plan file changes', category: 'Analysis', costTier: 'low' },
    ],
    'Web': [
        { name: 'web_search', description: 'Search the web', category: 'Web', costTier: 'medium' },
        { name: 'web_fetch', description: 'Fetch web content', category: 'Web', costTier: 'medium' },
    ],
    'Docker': [
        { name: 'docker_ps', description: 'List containers', category: 'Docker', costTier: 'free' },
        { name: 'docker_logs', description: 'View container logs', category: 'Docker', costTier: 'free' },
        { name: 'docker_exec', description: 'Execute in container', category: 'Docker', costTier: 'free' },
    ],
};

export class ToolsViewProvider implements vscode.TreeDataProvider<ToolItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<ToolItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    getTreeItem(element: ToolItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ToolItem): ToolItem[] {
        if (!element) {
            // Root level - show categories
            return Object.keys(TOOL_CATEGORIES).map(category => new ToolItem(
                category,
                undefined,
                vscode.TreeItemCollapsibleState.Collapsed,
                'category'
            ));
        }

        if (element.itemType === 'category') {
            // Show tools in category
            const tools = TOOL_CATEGORIES[element.label as string] || [];
            return tools.map(tool => new ToolItem(
                tool.name,
                tool,
                vscode.TreeItemCollapsibleState.None,
                'tool'
            ));
        }

        return [];
    }

    refresh(): void {
        this._onDidChangeTreeData.fire(undefined);
    }
}

class ToolItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly tool: ToolInfo | undefined,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly itemType: 'category' | 'tool'
    ) {
        super(label, collapsibleState);

        if (itemType === 'category') {
            this.iconPath = new vscode.ThemeIcon('folder');
            this.contextValue = 'category';
        } else if (tool) {
            this.description = tool.description;
            this.tooltip = `${tool.name}\n${tool.description}\nCost: ${tool.costTier}`;
            this.iconPath = new vscode.ThemeIcon(this._getIcon(tool));
            this.contextValue = 'tool';

            // Command to show tool details
            this.command = {
                command: 'victor.showToolInfo',
                title: 'Show Tool Info',
                arguments: [tool]
            };
        }
    }

    private _getIcon(tool: ToolInfo): string {
        const categoryIcons: Record<string, string> = {
            'File Operations': 'file',
            'Code Search': 'search',
            'Git': 'git-branch',
            'Shell': 'terminal',
            'Analysis': 'graph',
            'Web': 'globe',
            'Docker': 'package',
        };

        const costIcons: Record<string, string> = {
            'free': 'circle-outline',
            'low': 'circle-small',
            'medium': 'circle-large',
            'high': 'circle-filled',
        };

        return categoryIcons[tool.category] || 'tools';
    }
}

/**
 * Register tools view commands
 */
export function registerToolsCommands(
    context: vscode.ExtensionContext,
    provider: ToolsViewProvider
): void {
    // Show tool info
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showToolInfo', (tool: ToolInfo) => {
            const message = `**${tool.name}**\n\n${tool.description}\n\nCategory: ${tool.category}\nCost Tier: ${tool.costTier}`;
            vscode.window.showInformationMessage(message, { modal: true });
        })
    );

    // Refresh tools
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshTools', () => {
            provider.refresh();
        })
    );
}
