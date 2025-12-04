/**
 * File Watcher Provider
 *
 * Watches for file changes in the workspace and notifies relevant components.
 * Provides auto-refresh functionality for views and caches.
 */

import * as vscode from 'vscode';
import * as path from 'path';

export interface FileChangeEvent {
    type: 'create' | 'change' | 'delete';
    uri: vscode.Uri;
    relativePath: string;
    timestamp: Date;
}

export type FileChangeCallback = (event: FileChangeEvent) => void;

/**
 * Watches for file system changes in the workspace
 */
export class FileWatcher implements vscode.Disposable {
    private _watcher: vscode.FileSystemWatcher | null = null;
    private _callbacks: Set<FileChangeCallback> = new Set();
    private _debounceTimers: Map<string, NodeJS.Timeout> = new Map();
    private _recentChanges: FileChangeEvent[] = [];
    private _maxRecentChanges = 100;
    private _debounceMs = 300;
    private _disposables: vscode.Disposable[] = [];

    // Patterns to ignore
    private readonly _ignorePatterns = [
        '**/node_modules/**',
        '**/.git/**',
        '**/dist/**',
        '**/out/**',
        '**/.vscode/**',
        '**/__pycache__/**',
        '**/*.pyc',
        '**/venv/**',
        '**/.env/**',
        '**/coverage/**',
        '**/.nyc_output/**',
        '**/build/**',
        '**/.DS_Store',
        '**/Thumbs.db',
    ];

    constructor() {
        this._setupWatcher();
    }

    private _setupWatcher(): void {
        // Watch all files in workspace
        this._watcher = vscode.workspace.createFileSystemWatcher('**/*');

        this._watcher.onDidCreate((uri) => this._handleChange('create', uri));
        this._watcher.onDidChange((uri) => this._handleChange('change', uri));
        this._watcher.onDidDelete((uri) => this._handleChange('delete', uri));

        this._disposables.push(this._watcher);
    }

    private _handleChange(type: FileChangeEvent['type'], uri: vscode.Uri): void {
        // Check if file should be ignored
        if (this._shouldIgnore(uri)) {
            return;
        }

        const filePath = uri.fsPath;

        // Debounce rapid changes to the same file
        const existingTimer = this._debounceTimers.get(filePath);
        if (existingTimer) {
            clearTimeout(existingTimer);
        }

        const timer = setTimeout(() => {
            this._debounceTimers.delete(filePath);
            this._emitChange(type, uri);
        }, this._debounceMs);

        this._debounceTimers.set(filePath, timer);
    }

    private _emitChange(type: FileChangeEvent['type'], uri: vscode.Uri): void {
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        const relativePath = path.relative(workspaceRoot, uri.fsPath);

        const event: FileChangeEvent = {
            type,
            uri,
            relativePath,
            timestamp: new Date(),
        };

        // Store in recent changes
        this._recentChanges.unshift(event);
        if (this._recentChanges.length > this._maxRecentChanges) {
            this._recentChanges.pop();
        }

        // Notify all callbacks
        for (const callback of this._callbacks) {
            try {
                callback(event);
            } catch (error) {
                console.error('File watcher callback error:', error);
            }
        }
    }

    private _shouldIgnore(uri: vscode.Uri): boolean {
        const relativePath = vscode.workspace.asRelativePath(uri);

        for (const pattern of this._ignorePatterns) {
            // Simple glob matching
            const regex = this._globToRegex(pattern);
            if (regex.test(relativePath)) {
                return true;
            }
        }

        return false;
    }

    private _globToRegex(glob: string): RegExp {
        const escaped = glob
            .replace(/[.+^${}()|[\]\\]/g, '\\$&')
            .replace(/\*\*/g, '{{DOUBLE_STAR}}')
            .replace(/\*/g, '[^/]*')
            .replace(/\?/g, '.')
            .replace(/{{DOUBLE_STAR}}/g, '.*');

        return new RegExp(`^${escaped}$`);
    }

    /**
     * Register a callback for file changes
     */
    onFileChange(callback: FileChangeCallback): vscode.Disposable {
        this._callbacks.add(callback);
        return {
            dispose: () => {
                this._callbacks.delete(callback);
            },
        };
    }

    /**
     * Get recent file changes
     */
    getRecentChanges(count?: number): FileChangeEvent[] {
        return this._recentChanges.slice(0, count || this._maxRecentChanges);
    }

    /**
     * Get changes since a specific timestamp
     */
    getChangesSince(timestamp: Date): FileChangeEvent[] {
        return this._recentChanges.filter(
            (change) => change.timestamp > timestamp
        );
    }

    /**
     * Clear recent changes history
     */
    clearHistory(): void {
        this._recentChanges = [];
    }

    /**
     * Check if a specific file has changed recently
     */
    hasRecentChange(filePath: string, withinMs: number = 5000): boolean {
        const now = Date.now();
        return this._recentChanges.some(
            (change) =>
                (change.uri.fsPath === filePath || change.relativePath === filePath) &&
                now - change.timestamp.getTime() < withinMs
        );
    }

    /**
     * Set debounce interval
     */
    setDebounceMs(ms: number): void {
        this._debounceMs = ms;
    }

    /**
     * Add patterns to ignore
     */
    addIgnorePattern(pattern: string): void {
        if (!this._ignorePatterns.includes(pattern)) {
            this._ignorePatterns.push(pattern);
        }
    }

    /**
     * Remove patterns from ignore list
     */
    removeIgnorePattern(pattern: string): void {
        const index = this._ignorePatterns.indexOf(pattern);
        if (index !== -1) {
            this._ignorePatterns.splice(index, 1);
        }
    }

    dispose(): void {
        // Clear all debounce timers
        for (const timer of this._debounceTimers.values()) {
            clearTimeout(timer);
        }
        this._debounceTimers.clear();

        // Clear callbacks
        this._callbacks.clear();

        // Dispose watchers
        for (const disposable of this._disposables) {
            disposable.dispose();
        }
        this._disposables = [];
    }
}

/**
 * Integration helper for auto-refreshing views
 */
export class ViewRefreshManager {
    private _fileWatcher: FileWatcher;
    private _disposables: vscode.Disposable[] = [];
    private _refreshCallbacks: Map<string, () => void> = new Map();
    private _throttleMs = 1000;
    private _lastRefresh: Map<string, number> = new Map();

    constructor(fileWatcher: FileWatcher) {
        this._fileWatcher = fileWatcher;
        this._setupChangeHandler();
    }

    private _setupChangeHandler(): void {
        this._disposables.push(
            this._fileWatcher.onFileChange((event) => {
                this._handleFileChange(event);
            })
        );
    }

    private _handleFileChange(event: FileChangeEvent): void {
        const now = Date.now();

        // Refresh all registered views with throttling
        for (const [viewId, callback] of this._refreshCallbacks) {
            const lastRefresh = this._lastRefresh.get(viewId) || 0;
            if (now - lastRefresh >= this._throttleMs) {
                this._lastRefresh.set(viewId, now);
                try {
                    callback();
                } catch (error) {
                    console.error(`View refresh error for ${viewId}:`, error);
                }
            }
        }
    }

    /**
     * Register a view for auto-refresh on file changes
     */
    registerView(viewId: string, refreshCallback: () => void): vscode.Disposable {
        this._refreshCallbacks.set(viewId, refreshCallback);
        return {
            dispose: () => {
                this._refreshCallbacks.delete(viewId);
                this._lastRefresh.delete(viewId);
            },
        };
    }

    /**
     * Set throttle interval for view refreshes
     */
    setThrottleMs(ms: number): void {
        this._throttleMs = ms;
    }

    /**
     * Force refresh a specific view
     */
    forceRefresh(viewId: string): void {
        const callback = this._refreshCallbacks.get(viewId);
        if (callback) {
            this._lastRefresh.set(viewId, Date.now());
            callback();
        }
    }

    /**
     * Force refresh all views
     */
    forceRefreshAll(): void {
        const now = Date.now();
        for (const [viewId, callback] of this._refreshCallbacks) {
            this._lastRefresh.set(viewId, now);
            try {
                callback();
            } catch (error) {
                console.error(`View refresh error for ${viewId}:`, error);
            }
        }
    }

    dispose(): void {
        for (const disposable of this._disposables) {
            disposable.dispose();
        }
        this._disposables = [];
        this._refreshCallbacks.clear();
        this._lastRefresh.clear();
    }
}

/**
 * Create and register file watcher components
 */
export function createFileWatcher(context: vscode.ExtensionContext): {
    fileWatcher: FileWatcher;
    refreshManager: ViewRefreshManager;
} {
    const fileWatcher = new FileWatcher();
    const refreshManager = new ViewRefreshManager(fileWatcher);

    context.subscriptions.push(fileWatcher);
    context.subscriptions.push(refreshManager);

    // Register command to show recent changes
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showRecentChanges', async () => {
            const changes = fileWatcher.getRecentChanges(20);

            if (changes.length === 0) {
                vscode.window.showInformationMessage('No recent file changes');
                return;
            }

            const items = changes.map((change) => ({
                label: `${getChangeIcon(change.type)} ${change.relativePath}`,
                description: change.timestamp.toLocaleTimeString(),
                detail: change.type,
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Recent file changes',
            });

            if (selected) {
                // Find the change and open the file
                const change = changes.find(
                    (c) => c.relativePath === selected.label.substring(2).trim()
                );
                if (change && change.type !== 'delete') {
                    await vscode.window.showTextDocument(change.uri);
                }
            }
        })
    );

    return { fileWatcher, refreshManager };
}

function getChangeIcon(type: FileChangeEvent['type']): string {
    switch (type) {
        case 'create':
            return '$(new-file)';
        case 'change':
            return '$(edit)';
        case 'delete':
            return '$(trash)';
        default:
            return '$(file)';
    }
}
