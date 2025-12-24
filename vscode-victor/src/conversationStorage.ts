/**
 * Conversation Storage
 *
 * Persists chat history to disk for resuming conversations.
 * Uses JSON files with automatic backup and rotation.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { ChatMessage, ToolCall } from './victorClient';

export interface StoredConversation {
    id: string;
    title: string;
    messages: ChatMessage[];
    createdAt: number;
    updatedAt: number;
    workspaceFolder?: string;
    provider?: string;
    model?: string;
    metadata?: Record<string, unknown>;
}

export interface ConversationIndex {
    version: number;
    conversations: Array<{
        id: string;
        title: string;
        messageCount: number;
        createdAt: number;
        updatedAt: number;
        preview?: string;
    }>;
}

const STORAGE_VERSION = 1;
const MAX_CONVERSATIONS = 100;
const MAX_PREVIEW_LENGTH = 100;

export class ConversationStorage implements vscode.Disposable {
    private _disposables: vscode.Disposable[] = [];
    private _storageDir: string;
    private _index: ConversationIndex | null = null;
    private _currentConversation: StoredConversation | null = null;
    private _autoSaveInterval: NodeJS.Timeout | null = null;
    private _isDirty = false;

    constructor(
        private readonly _context: vscode.ExtensionContext,
        private readonly _log?: vscode.OutputChannel
    ) {
        // Create storage directory
        this._storageDir = path.join(_context.globalStorageUri.fsPath, 'conversations');
        this._ensureStorageDir();

        // Load index
        this._loadIndex();

        // Set up auto-save
        this._autoSaveInterval = setInterval(() => this._autoSave(), 30000); // Every 30 seconds

        // Register commands
        this._disposables.push(
            vscode.commands.registerCommand('victor.newConversation', () => this.newConversation()),
            vscode.commands.registerCommand('victor.loadConversation', () => this.showConversationPicker()),
            vscode.commands.registerCommand('victor.exportConversation', () => this.exportConversation()),
            vscode.commands.registerCommand('victor.deleteConversation', () => this.showDeletePicker()),
            vscode.commands.registerCommand('victor.clearAllConversations', () => this.clearAll()),
        );
    }

    public dispose(): void {
        if (this._autoSaveInterval) {
            clearInterval(this._autoSaveInterval);
        }
        this._save(); // Final save
        this._disposables.forEach(d => d.dispose());
    }

    /**
     * Get the current conversation
     */
    public get currentConversation(): StoredConversation | null {
        return this._currentConversation;
    }

    /**
     * Get all conversation summaries
     */
    public get conversations(): ConversationIndex['conversations'] {
        return this._index?.conversations || [];
    }

    /**
     * Create a new conversation
     */
    public newConversation(title?: string): StoredConversation {
        const conversation: StoredConversation = {
            id: this._generateId(),
            title: title || this._generateTitle(),
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
            workspaceFolder: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
        };

        this._currentConversation = conversation;
        this._isDirty = true;
        this._updateIndex(conversation);
        this._log?.appendLine(`[Storage] New conversation: ${conversation.id}`);

        return conversation;
    }

    /**
     * Add a message to the current conversation
     */
    public addMessage(message: ChatMessage): void {
        if (!this._currentConversation) {
            this.newConversation();
        }

        this._currentConversation!.messages.push(message);
        this._currentConversation!.updatedAt = Date.now();
        this._isDirty = true;

        // Update title from first user message if not set
        if (
            this._currentConversation!.title.startsWith('Conversation') &&
            message.role === 'user' &&
            this._currentConversation!.messages.filter(m => m.role === 'user').length === 1
        ) {
            this._currentConversation!.title = this._generateTitleFromMessage(message.content);
            this._updateIndex(this._currentConversation!);
        }
    }

    /**
     * Get messages for the current conversation
     */
    public getMessages(): ChatMessage[] {
        return this._currentConversation?.messages || [];
    }

    /**
     * Load a conversation by ID
     */
    public loadConversation(id: string): StoredConversation | null {
        const filePath = path.join(this._storageDir, `${id}.json`);

        try {
            if (fs.existsSync(filePath)) {
                const data = fs.readFileSync(filePath, 'utf-8');
                this._currentConversation = JSON.parse(data);
                this._log?.appendLine(`[Storage] Loaded conversation: ${id}`);
                return this._currentConversation;
            }
        } catch (error) {
            this._log?.appendLine(`[Storage] Failed to load ${id}: ${error}`);
        }

        return null;
    }

    /**
     * Show conversation picker
     */
    public async showConversationPicker(): Promise<void> {
        if (!this._index || this._index.conversations.length === 0) {
            vscode.window.showInformationMessage('No saved conversations');
            return;
        }

        const items = this._index.conversations.map(c => ({
            label: c.title,
            description: `${c.messageCount} messages`,
            detail: c.preview,
            id: c.id,
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a conversation to load',
        });

        if (selected) {
            this.loadConversation(selected.id);
            vscode.window.showInformationMessage(`Loaded: ${selected.label}`);
        }
    }

    /**
     * Export current conversation to markdown
     */
    public async exportConversation(): Promise<void> {
        if (!this._currentConversation || this._currentConversation.messages.length === 0) {
            vscode.window.showWarningMessage('No conversation to export');
            return;
        }

        const markdown = this._toMarkdown(this._currentConversation);

        // Let user choose save location
        const uri = await vscode.window.showSaveDialog({
            defaultUri: vscode.Uri.file(
                path.join(
                    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '',
                    `${this._currentConversation.title}.md`
                )
            ),
            filters: {
                'Markdown': ['md'],
                'JSON': ['json'],
            },
        });

        if (uri) {
            const content = uri.fsPath.endsWith('.json')
                ? JSON.stringify(this._currentConversation, null, 2)
                : markdown;

            fs.writeFileSync(uri.fsPath, content, 'utf-8');
            vscode.window.showInformationMessage(`Exported to ${uri.fsPath}`);
        }
    }

    /**
     * Show delete picker
     */
    public async showDeletePicker(): Promise<void> {
        if (!this._index || this._index.conversations.length === 0) {
            vscode.window.showInformationMessage('No saved conversations');
            return;
        }

        const items = this._index.conversations.map(c => ({
            label: c.title,
            description: new Date(c.updatedAt).toLocaleDateString(),
            id: c.id,
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select conversations to delete',
            canPickMany: true,
        });

        if (selected && selected.length > 0) {
            const confirm = await vscode.window.showWarningMessage(
                `Delete ${selected.length} conversation(s)?`,
                { modal: true },
                'Delete'
            );

            if (confirm === 'Delete') {
                for (const item of selected) {
                    this._deleteConversation(item.id);
                }
                vscode.window.showInformationMessage(`Deleted ${selected.length} conversation(s)`);
            }
        }
    }

    /**
     * Clear all conversations
     */
    public async clearAll(): Promise<void> {
        const confirm = await vscode.window.showWarningMessage(
            'Delete ALL saved conversations? This cannot be undone.',
            { modal: true },
            'Delete All'
        );

        if (confirm === 'Delete All') {
            // Delete all conversation files
            try {
                const files = fs.readdirSync(this._storageDir);
                for (const file of files) {
                    if (file.endsWith('.json')) {
                        fs.unlinkSync(path.join(this._storageDir, file));
                    }
                }

                // Reset index
                this._index = { version: STORAGE_VERSION, conversations: [] };
                this._saveIndex();
                this._currentConversation = null;

                vscode.window.showInformationMessage('All conversations deleted');
            } catch (error) {
                vscode.window.showErrorMessage(`Failed to clear: ${error}`);
            }
        }
    }

    /**
     * Force save current conversation
     */
    public save(): void {
        this._save();
    }

    // Private methods

    private _ensureStorageDir(): void {
        if (!fs.existsSync(this._storageDir)) {
            fs.mkdirSync(this._storageDir, { recursive: true });
        }
    }

    private _loadIndex(): void {
        const indexPath = path.join(this._storageDir, 'index.json');

        try {
            if (fs.existsSync(indexPath)) {
                const data = fs.readFileSync(indexPath, 'utf-8');
                this._index = JSON.parse(data);
            } else {
                this._index = { version: STORAGE_VERSION, conversations: [] };
            }
        } catch (error) {
            this._log?.appendLine(`[Storage] Failed to load index: ${error}`);
            this._index = { version: STORAGE_VERSION, conversations: [] };
        }
    }

    private _saveIndex(): void {
        const indexPath = path.join(this._storageDir, 'index.json');

        try {
            fs.writeFileSync(indexPath, JSON.stringify(this._index, null, 2), 'utf-8');
        } catch (error) {
            this._log?.appendLine(`[Storage] Failed to save index: ${error}`);
        }
    }

    private _updateIndex(conversation: StoredConversation): void {
        if (!this._index) {
            this._index = { version: STORAGE_VERSION, conversations: [] };
        }

        // Find existing entry
        const existingIndex = this._index.conversations.findIndex(c => c.id === conversation.id);
        const entry = {
            id: conversation.id,
            title: conversation.title,
            messageCount: conversation.messages.length,
            createdAt: conversation.createdAt,
            updatedAt: conversation.updatedAt,
            preview: this._getPreview(conversation),
        };

        if (existingIndex >= 0) {
            this._index.conversations[existingIndex] = entry;
        } else {
            this._index.conversations.unshift(entry);
        }

        // Sort by updatedAt
        this._index.conversations.sort((a, b) => b.updatedAt - a.updatedAt);

        // Trim old conversations
        if (this._index.conversations.length > MAX_CONVERSATIONS) {
            const removed = this._index.conversations.splice(MAX_CONVERSATIONS);
            for (const c of removed) {
                this._deleteConversation(c.id, false);
            }
        }

        this._saveIndex();
    }

    private _save(): void {
        if (!this._currentConversation || !this._isDirty) {
            return;
        }

        const filePath = path.join(this._storageDir, `${this._currentConversation.id}.json`);

        try {
            fs.writeFileSync(filePath, JSON.stringify(this._currentConversation, null, 2), 'utf-8');
            this._updateIndex(this._currentConversation);
            this._isDirty = false;
            this._log?.appendLine(`[Storage] Saved conversation: ${this._currentConversation.id}`);
        } catch (error) {
            this._log?.appendLine(`[Storage] Failed to save: ${error}`);
        }
    }

    private _autoSave(): void {
        if (this._isDirty) {
            this._save();
        }
    }

    private _deleteConversation(id: string, updateIndex = true): void {
        const filePath = path.join(this._storageDir, `${id}.json`);

        try {
            if (fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
            }

            if (updateIndex && this._index) {
                this._index.conversations = this._index.conversations.filter(c => c.id !== id);
                this._saveIndex();
            }

            if (this._currentConversation?.id === id) {
                this._currentConversation = null;
            }
        } catch (error) {
            this._log?.appendLine(`[Storage] Failed to delete ${id}: ${error}`);
        }
    }

    private _generateId(): string {
        return `conv-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    }

    private _generateTitle(): string {
        return `Conversation ${new Date().toLocaleDateString()}`;
    }

    private _generateTitleFromMessage(content: string): string {
        // Take first line or first 50 chars
        const firstLine = content.split('\n')[0];
        if (firstLine.length <= 50) {
            return firstLine;
        }
        return firstLine.slice(0, 47) + '...';
    }

    private _getPreview(conversation: StoredConversation): string {
        const lastUserMessage = [...conversation.messages]
            .reverse()
            .find(m => m.role === 'user');

        if (lastUserMessage) {
            const content = lastUserMessage.content;
            if (content.length <= MAX_PREVIEW_LENGTH) {
                return content;
            }
            return content.slice(0, MAX_PREVIEW_LENGTH - 3) + '...';
        }

        return '';
    }

    private _toMarkdown(conversation: StoredConversation): string {
        let md = `# ${conversation.title}\n\n`;
        md += `_Created: ${new Date(conversation.createdAt).toLocaleString()}_\n\n`;
        md += `---\n\n`;

        for (const message of conversation.messages) {
            const roleLabel = message.role === 'user' ? '**You**' : '**Victor**';
            md += `### ${roleLabel}\n\n`;
            md += message.content + '\n\n';

            if (message.toolCalls) {
                md += `<details>\n<summary>Tool Calls (${message.toolCalls.length})</summary>\n\n`;
                for (const tc of message.toolCalls) {
                    md += `- **${tc.name}**: ${tc.status}\n`;
                }
                md += `\n</details>\n\n`;
            }
        }

        return md;
    }
}
