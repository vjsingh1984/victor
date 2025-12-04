/**
 * Semantic Search Provider
 *
 * Provides semantic code search functionality powered by Victor's RAG system.
 */

import * as vscode from 'vscode';
import { VictorClient, SearchResult } from './victorClient';

export class SemanticSearchProvider {
    private _results: SearchResult[] = [];

    constructor(private readonly _client: VictorClient) {}

    async search(query: string): Promise<void> {
        const config = vscode.workspace.getConfiguration('victor');
        const maxResults = config.get('semanticSearch.maxResults', 10);

        // Show progress
        await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Searching codebase...',
                cancellable: true,
            },
            async (progress, token) => {
                try {
                    this._results = await this._client.semanticSearch(query, maxResults);

                    if (token.isCancellationRequested) {
                        return;
                    }

                    if (this._results.length === 0) {
                        vscode.window.showInformationMessage('No results found');
                        return;
                    }

                    // Show results in quick pick
                    await this._showResults(query);
                } catch (error) {
                    vscode.window.showErrorMessage(`Search failed: ${error}`);
                }
            }
        );
    }

    private async _showResults(query: string): Promise<void> {
        const items: vscode.QuickPickItem[] = this._results.map((result, index) => ({
            label: `$(file) ${this._getRelativePath(result.file)}`,
            description: `Line ${result.line}`,
            detail: this._truncate(result.content, 100),
            // Store index in alwaysShow to retrieve later
            alwaysShow: true,
        }));

        const selected = await vscode.window.showQuickPick(items, {
            title: `Search Results: "${query}"`,
            placeHolder: 'Select a result to open',
            matchOnDescription: true,
            matchOnDetail: true,
        });

        if (selected) {
            const index = items.indexOf(selected);
            if (index >= 0) {
                await this._openResult(this._results[index]);
            }
        }
    }

    private async _openResult(result: SearchResult): Promise<void> {
        try {
            const uri = vscode.Uri.file(result.file);
            const doc = await vscode.workspace.openTextDocument(uri);
            const editor = await vscode.window.showTextDocument(doc);

            // Go to line
            const line = Math.max(0, result.line - 1);
            const position = new vscode.Position(line, 0);
            editor.selection = new vscode.Selection(position, position);
            editor.revealRange(
                new vscode.Range(position, position),
                vscode.TextEditorRevealType.InCenter
            );

            // Highlight the line
            const lineRange = doc.lineAt(line).range;
            editor.setDecorations(this._getHighlightDecoration(), [lineRange]);

            // Clear highlight after 2 seconds
            setTimeout(() => {
                editor.setDecorations(this._getHighlightDecoration(), []);
            }, 2000);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to open file: ${error}`);
        }
    }

    private _highlightDecoration?: vscode.TextEditorDecorationType;

    private _getHighlightDecoration(): vscode.TextEditorDecorationType {
        if (!this._highlightDecoration) {
            this._highlightDecoration = vscode.window.createTextEditorDecorationType({
                backgroundColor: new vscode.ThemeColor('editor.findMatchHighlightBackground'),
                isWholeLine: true,
            });
        }
        return this._highlightDecoration;
    }

    private _getRelativePath(filePath: string): string {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders) {
            for (const folder of workspaceFolders) {
                if (filePath.startsWith(folder.uri.fsPath)) {
                    return filePath.substring(folder.uri.fsPath.length + 1);
                }
            }
        }
        return filePath;
    }

    private _truncate(text: string, maxLength: number): string {
        const cleaned = text.replace(/\s+/g, ' ').trim();
        if (cleaned.length <= maxLength) {
            return cleaned;
        }
        return cleaned.substring(0, maxLength - 3) + '...';
    }
}

/**
 * Search Results Tree View Provider
 *
 * Provides a tree view for browsing search results.
 */
export class SearchResultsTreeProvider implements vscode.TreeDataProvider<SearchResultItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<SearchResultItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private _results: SearchResult[] = [];
    private _query: string = '';

    setResults(results: SearchResult[], query: string): void {
        this._results = results;
        this._query = query;
        this._onDidChangeTreeData.fire(undefined);
    }

    clearResults(): void {
        this._results = [];
        this._query = '';
        this._onDidChangeTreeData.fire(undefined);
    }

    getTreeItem(element: SearchResultItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: SearchResultItem): SearchResultItem[] {
        if (!element) {
            // Root level - show results
            return this._results.map((result, index) => new SearchResultItem(result, index));
        }
        return [];
    }
}

class SearchResultItem extends vscode.TreeItem {
    constructor(
        public readonly result: SearchResult,
        public readonly index: number
    ) {
        super(
            `${result.file.split('/').pop()} (line ${result.line})`,
            vscode.TreeItemCollapsibleState.None
        );

        this.description = `Score: ${(result.score * 100).toFixed(1)}%`;
        this.tooltip = result.content;
        this.iconPath = new vscode.ThemeIcon('file-code');

        this.command = {
            command: 'vscode.open',
            title: 'Open',
            arguments: [
                vscode.Uri.file(result.file),
                {
                    selection: new vscode.Range(
                        result.line - 1, 0,
                        result.line - 1, 0
                    ),
                },
            ],
        };
    }
}
