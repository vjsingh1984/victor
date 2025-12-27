/**
 * Inline Completion Provider
 *
 * Provides AI-powered inline code completions as you type.
 */

import * as vscode from 'vscode';
import { VictorClient, CompletionRequest } from './victorClient';
import { TerminalHistoryService } from './terminalHistory';

export class InlineCompletionProvider implements vscode.InlineCompletionItemProvider, vscode.Disposable {
    private _debounceTimer: NodeJS.Timeout | null = null;
    private _abortController: AbortController | null = null;
    private _lastRequest: string = '';
    private _cache: Map<string, string[]> = new Map();
    private _maxCacheSize: number = 100;
    private _disposed: boolean = false;
    private _statusBarItem: vscode.StatusBarItem;
    private _isGenerating: boolean = false;

    constructor(private readonly _client: VictorClient) {
        // Create status bar item for completion indicator
        this._statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            95
        );
        this._statusBarItem.text = '$(loading~spin) Completing...';
        this._statusBarItem.tooltip = 'Victor AI is generating code completions';
    }

    dispose(): void {
        this._disposed = true;
        if (this._debounceTimer) {
            clearTimeout(this._debounceTimer);
            this._debounceTimer = null;
        }
        if (this._abortController) {
            this._abortController.abort();
            this._abortController = null;
        }
        this._cache.clear();
        this._statusBarItem.dispose();
    }

    private _showGenerating(): void {
        if (!this._isGenerating) {
            this._isGenerating = true;
            this._statusBarItem.show();
        }
    }

    private _hideGenerating(): void {
        if (this._isGenerating) {
            this._isGenerating = false;
            this._statusBarItem.hide();
        }
    }

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
        // Check if disposed
        if (this._disposed) {
            return null;
        }

        // Cancel any pending request
        if (this._abortController) {
            this._abortController.abort();
        }
        this._abortController = new AbortController();
        const signal = this._abortController.signal;

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
                suffix: suffix,  // FIM: include code after cursor for better context
                file: document.uri.fsPath,
                language: document.languageId,
                position: {
                    line: position.line,
                    character: position.character,
                },
                context: this._getFileContext(document, position),
                max_tokens: 128,  // Keep completions concise
                temperature: 0.0,  // Deterministic for inline suggestions
            };

            // Check cancellation before and after the request
            if (token.isCancellationRequested || signal.aborted) {
                return null;
            }

            // Show generating indicator
            this._showGenerating();

            try {
                const completions = await this._client.getCompletions(request, signal);

                if (token.isCancellationRequested || signal.aborted || completions.length === 0) {
                    return null;
                }

                // Cache results
                this._addToCache(cacheKey, completions);

                return completions.map(text => this._createCompletionItem(text, position));
            } finally {
                // Always hide the indicator
                this._hideGenerating();
            }
        } catch (error) {
            this._hideGenerating();
            // Ignore abort errors
            if (error instanceof Error && error.name === 'AbortError') {
                return null;
            }
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
        if (!match) {return '';}

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
        if (!match) {return '';}

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
     * Get context from @workspace (repo-aware context like Copilot)
     * Provides project structure, key files, and configuration
     */
    static async getWorkspaceContext(): Promise<string> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return '// No workspace open';
        }

        const lines: string[] = ['// Workspace Context:'];
        const rootPath = workspaceFolder.uri.fsPath;
        const rootName = workspaceFolder.name;

        lines.push(`// Project: ${rootName}`);

        // Find key configuration files
        const configFiles = [
            'package.json',
            'pyproject.toml',
            'Cargo.toml',
            'go.mod',
            'pom.xml',
            'build.gradle',
            'tsconfig.json',
            'setup.py',
            '.victor.md',
            'README.md',
        ];

        for (const config of configFiles) {
            try {
                const uri = vscode.Uri.joinPath(workspaceFolder.uri, config);
                const doc = await vscode.workspace.openTextDocument(uri);
                const content = doc.getText();

                // Truncate large files
                const truncated = content.length > 2000
                    ? content.slice(0, 2000) + '\n// ... (truncated)'
                    : content;

                lines.push(`\n// === ${config} ===`);
                lines.push(truncated);
            } catch {
                // File doesn't exist, skip
            }
        }

        // Get directory structure (top-level only)
        try {
            const files = await vscode.workspace.findFiles(
                '*',
                '{**/node_modules/**,**/.git/**,**/dist/**,**/build/**,**/__pycache__/**}',
                50
            );
            const dirs = await vscode.workspace.findFiles(
                '*/',
                '{**/node_modules/**,**/.git/**,**/dist/**,**/build/**,**/__pycache__/**}',
                20
            );

            lines.push('\n// === Directory Structure ===');
            for (const file of files) {
                const relativePath = vscode.workspace.asRelativePath(file);
                lines.push(`//   ${relativePath}`);
            }
        } catch {
            // Ignore errors
        }

        // Include open editors as context
        const openEditors = vscode.window.visibleTextEditors;
        if (openEditors.length > 0) {
            lines.push('\n// === Open Files ===');
            for (const editor of openEditors) {
                const relativePath = vscode.workspace.asRelativePath(editor.document.uri);
                lines.push(`//   ${relativePath} (${editor.document.languageId})`);
            }
        }

        return lines.join('\n');
    }

    /**
     * Get context from @selection (current editor selection)
     */
    static getSelectionContext(): string {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return '// No active editor';
        }

        const selection = editor.selection;
        if (selection.isEmpty) {
            return '// No selection';
        }

        const selectedText = editor.document.getText(selection);
        const relativePath = vscode.workspace.asRelativePath(editor.document.uri);
        const startLine = selection.start.line + 1;
        const endLine = selection.end.line + 1;

        return `// Selected from ${relativePath}:${startLine}-${endLine}\n${selectedText}`;
    }

    /**
     * Get context from @terminal (recent terminal output)
     */
    static getTerminalContext(): string {
        const terminal = vscode.window.activeTerminal;
        const historyService = TerminalHistoryService.getInstance();

        if (!terminal) {
            // No active terminal, but still show recent commands if available
            const recentContext = historyService.getContextString(5);
            if (recentContext !== '// No recent terminal commands') {
                return recentContext;
            }
            return '// No active terminal';
        }

        // Get commands from the specific terminal first, then global history
        const terminalCommands = historyService.getTerminalCommands(terminal.name, 5);

        if (terminalCommands.length > 0) {
            const lines = [`// Terminal: ${terminal.name}`, '// Recent commands:'];
            for (const entry of terminalCommands) {
                lines.push(`// $ ${entry.command}`);
                if (entry.exitCode !== undefined && entry.exitCode !== 0) {
                    lines.push(`//   (exit code: ${entry.exitCode})`);
                }
                if (entry.output) {
                    const outputLines = entry.output.split('\n').slice(0, 2);
                    for (const line of outputLines) {
                        lines.push(`//   > ${line.slice(0, 60)}`);
                    }
                }
            }
            return lines.join('\n');
        }

        // Fall back to global history
        return historyService.getContextString(5);
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
     * Supports: @workspace, @selection, @terminal, @problems, @git, @file:path, @folder:path
     */
    static async resolveAllMentions(message: string): Promise<string> {
        let resolved = message;

        // @workspace (repo-aware context like GitHub Copilot)
        if (message.includes('@workspace')) {
            const workspaceContext = await this.getWorkspaceContext();
            resolved = resolved.replace(/@workspace/g, workspaceContext);
        }

        // @selection (current editor selection)
        if (message.includes('@selection')) {
            const selectionContext = this.getSelectionContext();
            resolved = resolved.replace(/@selection/g, selectionContext);
        }

        // @terminal (terminal context)
        if (message.includes('@terminal')) {
            const terminalContext = this.getTerminalContext();
            resolved = resolved.replace(/@terminal/g, terminalContext);
        }

        // @problems
        if (message.includes('@problems')) {
            const problems = this.getProblemsContext();
            resolved = resolved.replace(/@problems/g, problems);
        }

        // @git
        if (message.includes('@git')) {
            const gitContext = await this.getGitContext();
            resolved = resolved.replace(/@git/g, gitContext);
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

    /**
     * Get list of available @ mention commands for autocomplete
     */
    static getAvailableMentions(): Array<{ label: string; description: string }> {
        return [
            { label: '@workspace', description: 'Project context (structure, configs, open files)' },
            { label: '@selection', description: 'Current editor selection' },
            { label: '@terminal', description: 'Active terminal context' },
            { label: '@problems', description: 'Workspace diagnostics (errors, warnings)' },
            { label: '@git', description: 'Git status (modified, staged files)' },
            { label: '@file:', description: 'Include specific file content' },
            { label: '@folder:', description: 'Include folder structure' },
        ];
    }
}
