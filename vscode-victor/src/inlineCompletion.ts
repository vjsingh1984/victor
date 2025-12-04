/**
 * Inline Completion Provider
 *
 * Provides AI-powered inline code completions similar to GitHub Copilot.
 */

import * as vscode from 'vscode';
import { VictorClient, CompletionRequest } from './victorClient';

export class InlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    private _debounceTimer: NodeJS.Timeout | null = null;
    private _lastRequest: string = '';
    private _cache: Map<string, string[]> = new Map();
    private _maxCacheSize: number = 100;

    constructor(private readonly _client: VictorClient) {}

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | null> {
        // Check if completions are enabled
        const config = vscode.workspace.getConfiguration('victor');
        if (!config.get('showInlineCompletions', true)) {
            return null;
        }

        // Don't trigger on automatic (typing) - only on explicit request
        // or after significant pause
        if (context.triggerKind === vscode.InlineCompletionTriggerKind.Automatic) {
            // Debounce for automatic triggers
            return this._debouncedCompletion(document, position, token);
        }

        return this._getCompletions(document, position, token);
    }

    private async _debouncedCompletion(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | null> {
        return new Promise((resolve) => {
            if (this._debounceTimer) {
                clearTimeout(this._debounceTimer);
            }

            this._debounceTimer = setTimeout(async () => {
                if (token.isCancellationRequested) {
                    resolve(null);
                    return;
                }

                const result = await this._getCompletions(document, position, token);
                resolve(result);
            }, 500); // 500ms debounce
        });
    }

    private async _getCompletions(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | null> {
        // Get context around cursor
        const prefix = this._getPrefix(document, position);
        const suffix = this._getSuffix(document, position);

        if (prefix.length < 3) {
            return null; // Not enough context
        }

        // Check cache
        const cacheKey = this._getCacheKey(document, position, prefix);
        if (this._cache.has(cacheKey)) {
            const cached = this._cache.get(cacheKey)!;
            return cached.map(text => this._createCompletionItem(text, position));
        }

        try {
            const request: CompletionRequest = {
                prompt: prefix,
                file: document.uri.fsPath,
                language: document.languageId,
                position: {
                    line: position.line,
                    character: position.character,
                },
                context: this._getFileContext(document, position),
            };

            const completions = await this._client.getCompletions(request);

            if (token.isCancellationRequested || completions.length === 0) {
                return null;
            }

            // Cache results
            this._addToCache(cacheKey, completions);

            return completions.map(text => this._createCompletionItem(text, position));
        } catch (error) {
            console.error('Completion error:', error);
            return null;
        }
    }

    private _createCompletionItem(
        text: string,
        position: vscode.Position
    ): vscode.InlineCompletionItem {
        return new vscode.InlineCompletionItem(
            text,
            new vscode.Range(position, position)
        );
    }

    private _getPrefix(document: vscode.TextDocument, position: vscode.Position): string {
        // Get up to 50 lines before cursor
        const startLine = Math.max(0, position.line - 50);
        const range = new vscode.Range(startLine, 0, position.line, position.character);
        return document.getText(range);
    }

    private _getSuffix(document: vscode.TextDocument, position: vscode.Position): string {
        // Get up to 10 lines after cursor
        const endLine = Math.min(document.lineCount - 1, position.line + 10);
        const range = new vscode.Range(
            position.line,
            position.character,
            endLine,
            document.lineAt(endLine).text.length
        );
        return document.getText(range);
    }

    private _getFileContext(document: vscode.TextDocument, position: vscode.Position): string {
        // Get imports and function signatures from the file
        const lines: string[] = [];
        const text = document.getText();
        const allLines = text.split('\n');

        for (let i = 0; i < Math.min(50, allLines.length); i++) {
            const line = allLines[i].trim();
            // Include imports, class definitions, function definitions
            if (
                line.startsWith('import ') ||
                line.startsWith('from ') ||
                line.startsWith('class ') ||
                line.startsWith('def ') ||
                line.startsWith('function ') ||
                line.startsWith('const ') ||
                line.startsWith('export ')
            ) {
                lines.push(allLines[i]);
            }
        }

        return lines.join('\n');
    }

    private _getCacheKey(
        document: vscode.TextDocument,
        position: vscode.Position,
        prefix: string
    ): string {
        // Create a cache key from document and position
        const lastLines = prefix.split('\n').slice(-3).join('\n');
        return `${document.uri.fsPath}:${position.line}:${lastLines.slice(-100)}`;
    }

    private _addToCache(key: string, completions: string[]): void {
        // LRU-style cache management
        if (this._cache.size >= this._maxCacheSize) {
            const firstKey = this._cache.keys().next().value;
            if (firstKey) {
                this._cache.delete(firstKey);
            }
        }
        this._cache.set(key, completions);
    }

    clearCache(): void {
        this._cache.clear();
    }
}

/**
 * Context Manager for gathering relevant code context
 *
 * Implements @file, @folder, @problems context providers
 */
export class ContextManager {
    /**
     * Get context from @file mentions
     */
    static async resolveFileContext(mention: string): Promise<string> {
        // Parse @file:path/to/file.ts
        const match = mention.match(/@file:(.+)/);
        if (!match) return '';

        const filePath = match[1].trim();
        try {
            const uri = vscode.Uri.file(filePath);
            const doc = await vscode.workspace.openTextDocument(uri);
            return `// File: ${filePath}\n${doc.getText()}`;
        } catch {
            return `// File not found: ${filePath}`;
        }
    }

    /**
     * Get context from @folder mentions
     */
    static async resolveFolderContext(mention: string): Promise<string> {
        const match = mention.match(/@folder:(.+)/);
        if (!match) return '';

        const folderPath = match[1].trim();
        try {
            const uri = vscode.Uri.file(folderPath);
            const files = await vscode.workspace.findFiles(
                new vscode.RelativePattern(uri, '**/*'),
                '**/node_modules/**',
                50 // Max files
            );

            const tree: string[] = [`// Folder: ${folderPath}`];
            for (const file of files) {
                const relativePath = vscode.workspace.asRelativePath(file);
                tree.push(`  ${relativePath}`);
            }
            return tree.join('\n');
        } catch {
            return `// Folder not found: ${folderPath}`;
        }
    }

    /**
     * Get context from @problems (workspace diagnostics)
     */
    static getProblemsContext(): string {
        const allDiagnostics: string[] = ['// Current Problems:'];

        for (const [uri, diagnostics] of vscode.languages.getDiagnostics()) {
            const relativePath = vscode.workspace.asRelativePath(uri);
            for (const diag of diagnostics) {
                if (diag.severity === vscode.DiagnosticSeverity.Error ||
                    diag.severity === vscode.DiagnosticSeverity.Warning) {
                    const severity = diag.severity === vscode.DiagnosticSeverity.Error ? 'ERROR' : 'WARNING';
                    allDiagnostics.push(
                        `// ${relativePath}:${diag.range.start.line + 1} [${severity}] ${diag.message}`
                    );
                }
            }
        }

        return allDiagnostics.length > 1 ? allDiagnostics.join('\n') : '// No problems found';
    }

    /**
     * Get context from @git (recent changes)
     */
    static async getGitContext(): Promise<string> {
        try {
            const gitExtension = vscode.extensions.getExtension('vscode.git');
            if (!gitExtension) {
                return '// Git extension not available';
            }

            const git = gitExtension.exports.getAPI(1);
            const repo = git.repositories[0];
            if (!repo) {
                return '// No git repository found';
            }

            const changes: string[] = ['// Git Status:'];
            const status = repo.state;

            for (const change of status.workingTreeChanges) {
                const path = vscode.workspace.asRelativePath(change.uri);
                changes.push(`// Modified: ${path}`);
            }

            for (const change of status.indexChanges) {
                const path = vscode.workspace.asRelativePath(change.uri);
                changes.push(`// Staged: ${path}`);
            }

            return changes.length > 1 ? changes.join('\n') : '// No changes';
        } catch {
            return '// Git context unavailable';
        }
    }

    /**
     * Resolve all @ mentions in a message
     */
    static async resolveAllMentions(message: string): Promise<string> {
        let resolved = message;

        // @problems
        if (message.includes('@problems')) {
            const problems = this.getProblemsContext();
            resolved = resolved.replace('@problems', problems);
        }

        // @git
        if (message.includes('@git')) {
            const gitContext = await this.getGitContext();
            resolved = resolved.replace('@git', gitContext);
        }

        // @file:path
        const fileMatches = message.matchAll(/@file:[^\s]+/g);
        for (const match of fileMatches) {
            const fileContext = await this.resolveFileContext(match[0]);
            resolved = resolved.replace(match[0], fileContext);
        }

        // @folder:path
        const folderMatches = message.matchAll(/@folder:[^\s]+/g);
        for (const match of folderMatches) {
            const folderContext = await this.resolveFolderContext(match[0]);
            resolved = resolved.replace(match[0], folderContext);
        }

        return resolved;
    }
}
