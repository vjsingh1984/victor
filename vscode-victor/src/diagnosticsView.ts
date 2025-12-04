/**
 * Diagnostics View Provider
 *
 * Provides a tree view of diagnostics (errors/warnings) with AI-powered actions:
 * - Shows all workspace diagnostics organized by file
 * - Quick fix suggestions via Victor AI
 * - Batch fix capabilities
 */

import * as vscode from 'vscode';
import * as path from 'path';

interface DiagnosticInfo {
    uri: vscode.Uri;
    diagnostic: vscode.Diagnostic;
}

/**
 * Diagnostics View Provider
 */
export class DiagnosticsViewProvider implements vscode.TreeDataProvider<DiagnosticItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<DiagnosticItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private _disposables: vscode.Disposable[] = [];
    private _groupByFile = true;

    constructor() {
        // Listen for diagnostic changes
        this._disposables.push(
            vscode.languages.onDidChangeDiagnostics(() => {
                this.refresh();
            })
        );
    }

    refresh(): void {
        this._onDidChangeTreeData.fire(undefined);
    }

    setGroupByFile(groupByFile: boolean): void {
        this._groupByFile = groupByFile;
        this.refresh();
    }

    getTreeItem(element: DiagnosticItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: DiagnosticItem): DiagnosticItem[] {
        if (!element) {
            // Root level
            if (this._groupByFile) {
                return this._getFileItems();
            } else {
                return this._getAllDiagnostics();
            }
        }

        // Children of file item
        if (element.itemType === 'file' && element.uri) {
            return this._getDiagnosticsForFile(element.uri);
        }

        return [];
    }

    private _getFileItems(): DiagnosticItem[] {
        const allDiagnostics = vscode.languages.getDiagnostics();
        const items: DiagnosticItem[] = [];

        for (const [uri, diagnostics] of allDiagnostics) {
            if (diagnostics.length === 0) continue;

            const errorCount = diagnostics.filter(
                d => d.severity === vscode.DiagnosticSeverity.Error
            ).length;
            const warningCount = diagnostics.filter(
                d => d.severity === vscode.DiagnosticSeverity.Warning
            ).length;

            items.push(new DiagnosticItem(
                path.basename(uri.fsPath),
                'file',
                vscode.TreeItemCollapsibleState.Expanded,
                {
                    uri,
                    description: `${errorCount} errors, ${warningCount} warnings`,
                    tooltip: uri.fsPath,
                }
            ));
        }

        if (items.length === 0) {
            return [new DiagnosticItem(
                'No problems found',
                'empty',
                vscode.TreeItemCollapsibleState.None,
                { icon: 'check' }
            )];
        }

        // Sort by error count descending
        return items.sort((a, b) => {
            const aDesc = typeof a.description === 'string' ? a.description : '';
            const bDesc = typeof b.description === 'string' ? b.description : '';
            const aErrors = parseInt(aDesc.split(' ')[0] || '0');
            const bErrors = parseInt(bDesc.split(' ')[0] || '0');
            return bErrors - aErrors;
        });
    }

    private _getDiagnosticsForFile(uri: vscode.Uri): DiagnosticItem[] {
        const diagnostics = vscode.languages.getDiagnostics(uri);
        return diagnostics.map(d => this._createDiagnosticItem(uri, d));
    }

    private _getAllDiagnostics(): DiagnosticItem[] {
        const allDiagnostics = vscode.languages.getDiagnostics();
        const items: DiagnosticItem[] = [];

        for (const [uri, diagnostics] of allDiagnostics) {
            for (const diagnostic of diagnostics) {
                items.push(this._createDiagnosticItem(uri, diagnostic));
            }
        }

        if (items.length === 0) {
            return [new DiagnosticItem(
                'No problems found',
                'empty',
                vscode.TreeItemCollapsibleState.None,
                { icon: 'check' }
            )];
        }

        // Sort by severity (errors first)
        return items.sort((a, b) => {
            const aSeverity = a.diagnostic?.severity ?? 3;
            const bSeverity = b.diagnostic?.severity ?? 3;
            return aSeverity - bSeverity;
        });
    }

    private _createDiagnosticItem(uri: vscode.Uri, diagnostic: vscode.Diagnostic): DiagnosticItem {
        const line = diagnostic.range.start.line + 1;
        const severity = this._getSeverityLabel(diagnostic.severity);
        const message = diagnostic.message.split('\n')[0]; // First line only

        return new DiagnosticItem(
            message,
            'diagnostic',
            vscode.TreeItemCollapsibleState.None,
            {
                uri,
                diagnostic,
                description: `Line ${line}`,
                tooltip: `${severity}: ${diagnostic.message}\n\nSource: ${diagnostic.source || 'Unknown'}`,
                icon: this._getSeverityIcon(diagnostic.severity),
            }
        );
    }

    private _getSeverityLabel(severity: vscode.DiagnosticSeverity): string {
        switch (severity) {
            case vscode.DiagnosticSeverity.Error:
                return 'Error';
            case vscode.DiagnosticSeverity.Warning:
                return 'Warning';
            case vscode.DiagnosticSeverity.Information:
                return 'Info';
            case vscode.DiagnosticSeverity.Hint:
                return 'Hint';
            default:
                return 'Unknown';
        }
    }

    private _getSeverityIcon(severity: vscode.DiagnosticSeverity): string {
        switch (severity) {
            case vscode.DiagnosticSeverity.Error:
                return 'error';
            case vscode.DiagnosticSeverity.Warning:
                return 'warning';
            case vscode.DiagnosticSeverity.Information:
                return 'info';
            case vscode.DiagnosticSeverity.Hint:
                return 'lightbulb';
            default:
                return 'circle-outline';
        }
    }

    dispose(): void {
        for (const d of this._disposables) {
            d.dispose();
        }
    }
}

class DiagnosticItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly itemType: 'file' | 'diagnostic' | 'empty',
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        options: {
            uri?: vscode.Uri;
            diagnostic?: vscode.Diagnostic;
            description?: string;
            tooltip?: string;
            icon?: string;
        } = {}
    ) {
        super(label, collapsibleState);

        this.uri = options.uri;
        this.diagnostic = options.diagnostic;
        this.description = options.description;
        this.tooltip = options.tooltip;

        if (options.icon) {
            this.iconPath = new vscode.ThemeIcon(options.icon);
        }

        if (itemType === 'file') {
            this.iconPath = new vscode.ThemeIcon('file');
            this.contextValue = 'diagnosticFile';
        } else if (itemType === 'diagnostic' && options.uri && options.diagnostic) {
            this.contextValue = 'diagnostic';
            this.command = {
                command: 'victor.goToDiagnostic',
                title: 'Go to Diagnostic',
                arguments: [options.uri, options.diagnostic],
            };
        }
    }

    public readonly uri?: vscode.Uri;
    public readonly diagnostic?: vscode.Diagnostic;
}

/**
 * Register diagnostics commands
 */
export function registerDiagnosticsCommands(
    context: vscode.ExtensionContext,
    provider: DiagnosticsViewProvider,
    sendToChat: (message: string) => Promise<void>
): void {
    // Go to diagnostic
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.goToDiagnostic',
            async (uri: vscode.Uri, diagnostic: vscode.Diagnostic) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const editor = await vscode.window.showTextDocument(document);
                editor.selection = new vscode.Selection(
                    diagnostic.range.start,
                    diagnostic.range.end
                );
                editor.revealRange(diagnostic.range, vscode.TextEditorRevealType.InCenter);
            }
        )
    );

    // Fix diagnostic with Victor
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.fixDiagnosticFromView',
            async (item: DiagnosticItem) => {
                if (!item.uri || !item.diagnostic) return;

                const document = await vscode.workspace.openTextDocument(item.uri);
                const code = document.getText(item.diagnostic.range);

                const prompt = `Fix this ${document.languageId} error:

Error: ${item.diagnostic.message}
File: ${path.basename(item.uri.fsPath)}
Line: ${item.diagnostic.range.start.line + 1}

Code:
\`\`\`${document.languageId}
${code}
\`\`\`

Provide the corrected code.`;

                await sendToChat(prompt);
            }
        )
    );

    // Fix all diagnostics in file
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.fixAllInFile',
            async (item: DiagnosticItem) => {
                if (!item.uri) return;

                const document = await vscode.workspace.openTextDocument(item.uri);
                const diagnostics = vscode.languages.getDiagnostics(item.uri);

                // Focus on errors first
                const errors = diagnostics.filter(
                    d => d.severity === vscode.DiagnosticSeverity.Error
                );

                if (errors.length === 0) {
                    vscode.window.showInformationMessage('No errors to fix');
                    return;
                }

                const diagnosticList = errors.map(d => {
                    const line = d.range.start.line + 1;
                    return `- Line ${line}: ${d.message}`;
                }).join('\n');

                const prompt = `Fix these ${document.languageId} errors in ${path.basename(item.uri.fsPath)}:

${diagnosticList}

Full file content:
\`\`\`${document.languageId}
${document.getText()}
\`\`\`

Provide the corrected file content.`;

                await sendToChat(prompt);
            }
        )
    );

    // Explain all diagnostics
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.explainAllDiagnostics',
            async () => {
                const allDiagnostics = vscode.languages.getDiagnostics();
                const items: DiagnosticInfo[] = [];

                for (const [uri, diagnostics] of allDiagnostics) {
                    for (const diagnostic of diagnostics) {
                        items.push({ uri, diagnostic });
                    }
                }

                if (items.length === 0) {
                    vscode.window.showInformationMessage('No diagnostics to explain');
                    return;
                }

                const summary = items.slice(0, 10).map(item => {
                    const severity = item.diagnostic.severity === vscode.DiagnosticSeverity.Error
                        ? 'ERROR'
                        : 'WARNING';
                    return `[${severity}] ${path.basename(item.uri.fsPath)}:${item.diagnostic.range.start.line + 1} - ${item.diagnostic.message}`;
                }).join('\n');

                const prompt = `Explain these ${items.length} diagnostics and suggest how to fix them:

${summary}
${items.length > 10 ? `\n... and ${items.length - 10} more` : ''}

Provide a summary of the issues and recommended fixes.`;

                await sendToChat(prompt);
            }
        )
    );

    // Refresh diagnostics
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshDiagnostics', () => {
            provider.refresh();
        })
    );

    // Toggle group by file
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.toggleDiagnosticsGrouping', () => {
            // Toggle would need state management
            provider.refresh();
        })
    );
}
