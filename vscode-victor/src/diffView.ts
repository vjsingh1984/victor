/**
 * Diff View Provider
 *
 * Provides inline diff preview and application UI for proposed file changes.
 * Similar to how Claude Code shows file modifications before applying them.
 */

import * as vscode from 'vscode';
import * as path from 'path';

export interface FileChange {
    filePath: string;
    originalContent: string;
    newContent: string;
    changeType: 'create' | 'modify' | 'delete';
    description?: string;
}

export interface DiffSession {
    id: string;
    changes: FileChange[];
    timestamp: Date;
    description: string;
}

/**
 * Manages diff previews and file change applications
 */
export class DiffViewProvider {
    private _pendingSessions: Map<string, DiffSession> = new Map();
    private _decorationType: vscode.TextEditorDecorationType;
    private _statusBarItem: vscode.StatusBarItem;

    constructor() {
        // Create decoration type for highlighting changes
        this._decorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('diffEditor.insertedTextBackground'),
            isWholeLine: true,
        });

        // Create status bar item for pending changes
        this._statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            98
        );
        this._statusBarItem.command = 'victor.showPendingChanges';
        this._updateStatusBar();
    }

    /**
     * Create a new diff session with proposed changes
     */
    createSession(changes: FileChange[], description: string): string {
        const sessionId = `diff-${Date.now()}`;
        const session: DiffSession = {
            id: sessionId,
            changes,
            timestamp: new Date(),
            description,
        };
        this._pendingSessions.set(sessionId, session);
        this._updateStatusBar();
        return sessionId;
    }

    /**
     * Show diff preview for a specific file change
     */
    async showDiff(change: FileChange): Promise<void> {
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        const absolutePath = path.isAbsolute(change.filePath)
            ? change.filePath
            : path.join(workspaceRoot, change.filePath);

        // Create URIs for diff view
        const originalUri = vscode.Uri.parse(`victor-original:${change.filePath}`);
        const modifiedUri = vscode.Uri.file(absolutePath);

        // For new files, show the new content directly
        if (change.changeType === 'create') {
            const doc = await vscode.workspace.openTextDocument({
                content: change.newContent,
                language: this._getLanguageId(change.filePath),
            });
            await vscode.window.showTextDocument(doc, { preview: true });
            return;
        }

        // For modifications, show side-by-side diff
        const title = `${path.basename(change.filePath)} (Proposed Changes)`;
        await vscode.commands.executeCommand(
            'vscode.diff',
            originalUri,
            modifiedUri,
            title,
            { preview: true }
        );
    }

    /**
     * Show all pending changes in a quick pick
     */
    async showPendingChanges(): Promise<void> {
        if (this._pendingSessions.size === 0) {
            vscode.window.showInformationMessage('No pending changes');
            return;
        }

        const items: vscode.QuickPickItem[] = [];

        for (const [sessionId, session] of this._pendingSessions) {
            items.push({
                label: `$(git-commit) ${session.description}`,
                description: `${session.changes.length} file(s)`,
                detail: session.changes.map(c => `  ${this._getChangeIcon(c.changeType)} ${c.filePath}`).join('\n'),
            });

            // Add individual file items
            for (const change of session.changes) {
                items.push({
                    label: `    ${this._getChangeIcon(change.changeType)} ${path.basename(change.filePath)}`,
                    description: change.filePath,
                    detail: change.description,
                });
            }
        }

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a change to preview',
            matchOnDescription: true,
        });

        if (selected?.description) {
            // Find the change and show diff
            for (const session of this._pendingSessions.values()) {
                const change = session.changes.find(c => c.filePath === selected.description);
                if (change) {
                    await this.showDiff(change);
                    break;
                }
            }
        }
    }

    /**
     * Apply a single file change
     */
    async applyChange(change: FileChange): Promise<boolean> {
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        const absolutePath = path.isAbsolute(change.filePath)
            ? change.filePath
            : path.join(workspaceRoot, change.filePath);

        try {
            const uri = vscode.Uri.file(absolutePath);

            switch (change.changeType) {
                case 'create':
                case 'modify':
                    const edit = new vscode.WorkspaceEdit();
                    if (change.changeType === 'create') {
                        edit.createFile(uri, { overwrite: false, ignoreIfExists: false });
                    }
                    edit.replace(
                        uri,
                        new vscode.Range(0, 0, Number.MAX_VALUE, 0),
                        change.newContent
                    );
                    await vscode.workspace.applyEdit(edit);
                    break;

                case 'delete':
                    await vscode.workspace.fs.delete(uri);
                    break;
            }

            return true;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to apply change to ${change.filePath}: ${error}`);
            return false;
        }
    }

    /**
     * Apply all changes in a session
     */
    async applySession(sessionId: string): Promise<boolean> {
        const session = this._pendingSessions.get(sessionId);
        if (!session) {
            vscode.window.showErrorMessage('Session not found');
            return false;
        }

        const confirm = await vscode.window.showWarningMessage(
            `Apply ${session.changes.length} change(s)?`,
            { modal: true },
            'Apply All',
            'Review First'
        );

        if (confirm === 'Review First') {
            await this.showPendingChanges();
            return false;
        }

        if (confirm !== 'Apply All') {
            return false;
        }

        let successCount = 0;
        for (const change of session.changes) {
            if (await this.applyChange(change)) {
                successCount++;
            }
        }

        if (successCount === session.changes.length) {
            vscode.window.showInformationMessage(`Applied all ${successCount} changes`);
            this._pendingSessions.delete(sessionId);
            this._updateStatusBar();
            return true;
        } else {
            vscode.window.showWarningMessage(
                `Applied ${successCount}/${session.changes.length} changes`
            );
            return false;
        }
    }

    /**
     * Reject/discard a session
     */
    rejectSession(sessionId: string): void {
        if (this._pendingSessions.delete(sessionId)) {
            this._updateStatusBar();
            vscode.window.showInformationMessage('Changes discarded');
        }
    }

    /**
     * Get all pending sessions
     */
    getPendingSessions(): DiffSession[] {
        return Array.from(this._pendingSessions.values());
    }

    /**
     * Parse file changes from AI response
     */
    parseChangesFromResponse(response: string): FileChange[] {
        const changes: FileChange[] = [];

        // Match code blocks with file paths
        const codeBlockRegex = /```(?:(\w+)\s+)?(?:([^\n]+\.[\w]+)\n)?([\s\S]*?)```/g;
        let match;

        while ((match = codeBlockRegex.exec(response)) !== null) {
            const language = match[1];
            const filePath = match[2];
            const content = match[3]?.trim();

            if (filePath && content) {
                changes.push({
                    filePath,
                    originalContent: '', // Will be fetched when applying
                    newContent: content,
                    changeType: 'modify', // Default to modify
                    description: `${language || 'code'} changes`,
                });
            }
        }

        // Match diff blocks
        const diffRegex = /```diff\n([\s\S]*?)```/g;
        while ((match = diffRegex.exec(response)) !== null) {
            const diffContent = match[1];
            const fileMatch = diffContent.match(/^(?:---|\+\+\+)\s+([^\n]+)/m);
            if (fileMatch) {
                changes.push({
                    filePath: fileMatch[1].replace(/^[ab]\//, ''),
                    originalContent: '',
                    newContent: diffContent,
                    changeType: 'modify',
                    description: 'Diff patch',
                });
            }
        }

        return changes;
    }

    private _getLanguageId(filePath: string): string {
        const ext = path.extname(filePath).toLowerCase();
        const languageMap: Record<string, string> = {
            '.ts': 'typescript',
            '.tsx': 'typescriptreact',
            '.js': 'javascript',
            '.jsx': 'javascriptreact',
            '.py': 'python',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sql': 'sql',
        };
        return languageMap[ext] || 'plaintext';
    }

    private _getChangeIcon(changeType: FileChange['changeType']): string {
        switch (changeType) {
            case 'create': return '$(new-file)';
            case 'modify': return '$(edit)';
            case 'delete': return '$(trash)';
            default: return '$(file)';
        }
    }

    private _updateStatusBar(): void {
        const totalChanges = Array.from(this._pendingSessions.values())
            .reduce((sum, s) => sum + s.changes.length, 0);

        if (totalChanges > 0) {
            this._statusBarItem.text = `$(git-compare) ${totalChanges} pending`;
            this._statusBarItem.tooltip = `${totalChanges} pending file change(s)\nClick to review`;
            this._statusBarItem.show();
        } else {
            this._statusBarItem.hide();
        }
    }

    dispose(): void {
        this._decorationType.dispose();
        this._statusBarItem.dispose();
    }
}

/**
 * Content provider for original file content in diffs
 */
export class OriginalContentProvider implements vscode.TextDocumentContentProvider {
    private _originalContents: Map<string, string> = new Map();

    provideTextDocumentContent(uri: vscode.Uri): string {
        return this._originalContents.get(uri.path) || '';
    }

    setOriginalContent(filePath: string, content: string): void {
        this._originalContents.set(filePath, content);
    }

    clearOriginalContent(filePath: string): void {
        this._originalContents.delete(filePath);
    }
}

/**
 * Register diff view commands
 */
export function registerDiffCommands(
    context: vscode.ExtensionContext,
    diffProvider: DiffViewProvider
): void {
    // Show pending changes
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showPendingChanges', () => {
            diffProvider.showPendingChanges();
        })
    );

    // Apply all pending changes
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.applyAllChanges', async () => {
            const sessions = diffProvider.getPendingSessions();
            if (sessions.length === 0) {
                vscode.window.showInformationMessage('No pending changes to apply');
                return;
            }

            for (const session of sessions) {
                await diffProvider.applySession(session.id);
            }
        })
    );

    // Reject all pending changes
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.rejectAllChanges', () => {
            const sessions = diffProvider.getPendingSessions();
            if (sessions.length === 0) {
                vscode.window.showInformationMessage('No pending changes to reject');
                return;
            }

            for (const session of sessions) {
                diffProvider.rejectSession(session.id);
            }
        })
    );

    // Register content provider for original files
    const contentProvider = new OriginalContentProvider();
    context.subscriptions.push(
        vscode.workspace.registerTextDocumentContentProvider('victor-original', contentProvider)
    );
}
