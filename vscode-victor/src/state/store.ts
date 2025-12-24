/**
 * State Store
 *
 * Centralized state management for the Victor VS Code extension.
 * Provides:
 * - Type-safe state access and updates
 * - Event-based change notifications
 * - Persistence to VS Code globalState
 * - Selector-based subscriptions
 */

import * as vscode from 'vscode';
import { ServerStatus } from '../serverManager';
import {
    AppState,
    AgentMode,
    ConnectionState,
    ConversationState,
    ContextItem,
    ModelInfo,
    ProviderType,
    ServerState,
    SessionState,
    SettingsState,
    UIState,
    StateChangeEvent,
    StateSubscriber,
    StateSelector,
    DeepPartial,
} from './types';

/**
 * Generate a unique ID
 */
function generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Deep merge utility
 */
function deepMerge<T extends object>(target: T, source: DeepPartial<T>): T {
    const result = { ...target };

    for (const key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
            const sourceValue = source[key];
            const targetValue = target[key];

            if (
                sourceValue !== null &&
                typeof sourceValue === 'object' &&
                !Array.isArray(sourceValue) &&
                targetValue !== null &&
                typeof targetValue === 'object' &&
                !Array.isArray(targetValue)
            ) {
                (result as Record<string, unknown>)[key] = deepMerge(
                    targetValue as object,
                    sourceValue as DeepPartial<typeof targetValue>
                );
            } else {
                (result as Record<string, unknown>)[key] = sourceValue;
            }
        }
    }

    return result;
}

/**
 * Create initial state
 */
function createInitialState(): AppState {
    return {
        server: {
            status: ServerStatus.Stopped,
            url: 'http://localhost:8000',
            port: 8000,
            connectionState: ConnectionState.Disconnected,
        },
        session: {
            mode: 'build',
            model: {
                provider: 'anthropic',
                modelId: 'claude-sonnet-4-20250514',
                displayName: 'Claude Sonnet 4',
            },
            conversation: {
                id: generateId(),
                messages: [],
                isStreaming: false,
                lastActivity: Date.now(),
            },
            context: [],
        },
        ui: {
            chatViewVisible: false,
            sidebarOpen: true,
            activePanel: null,
            statusBarMode: 'build',
        },
        settings: {
            autoStart: false,
            showInlineCompletions: true,
            semanticSearchEnabled: true,
            semanticSearchMaxResults: 10,
            serverPort: 8000,
            serverApiKey: '',
        },
        initialized: false,
    };
}

/**
 * State Store Class
 *
 * Singleton pattern for centralized state management.
 */
export class StateStore {
    private static instance: StateStore | null = null;

    private state: AppState;
    private subscribers: Map<string, StateSubscriber[]> = new Map();
    private selectorSubscribers: Map<StateSelector<unknown>, Set<StateSubscriber<unknown>>> = new Map();
    private globalState: vscode.Memento | null = null;
    private readonly STORAGE_KEY = 'victor.state';

    private _onStateChange = new vscode.EventEmitter<StateChangeEvent>();
    readonly onStateChange = this._onStateChange.event;

    private constructor() {
        this.state = createInitialState();
    }

    /**
     * Get the singleton instance
     */
    static getInstance(): StateStore {
        if (!StateStore.instance) {
            StateStore.instance = new StateStore();
        }
        return StateStore.instance;
    }

    /**
     * Reset instance (for testing)
     */
    static resetInstance(): void {
        StateStore.instance = null;
    }

    /**
     * Initialize the store with VS Code context
     */
    async initialize(context: vscode.ExtensionContext): Promise<void> {
        this.globalState = context.globalState;

        // Load persisted state
        const persistedState = this.globalState.get<Partial<AppState>>(this.STORAGE_KEY);
        if (persistedState) {
            // Merge persisted state with defaults (to handle new fields)
            this.state = deepMerge(createInitialState(), persistedState);
        }

        // Sync with VS Code configuration
        await this.syncWithConfiguration();

        // Listen for configuration changes
        context.subscriptions.push(
            vscode.workspace.onDidChangeConfiguration(e => {
                if (e.affectsConfiguration('victor')) {
                    this.syncWithConfiguration();
                }
            })
        );

        this.state.initialized = true;
    }

    /**
     * Sync state with VS Code configuration
     */
    private async syncWithConfiguration(): Promise<void> {
        const config = vscode.workspace.getConfiguration('victor');

        this.updateState({
            server: {
                port: config.get('serverPort', 8000),
                url: `http://localhost:${config.get('serverPort', 8000)}`,
            },
            session: {
                mode: config.get('mode', 'build') as AgentMode,
                model: {
                    provider: config.get('provider', 'anthropic') as ProviderType,
                    modelId: config.get('model', 'claude-sonnet-4-20250514'),
                    displayName: this.getModelDisplayName(
                        config.get('provider', 'anthropic') as ProviderType,
                        config.get('model', 'claude-sonnet-4-20250514')
                    ),
                },
            },
            settings: {
                autoStart: config.get('autoStart', false),
                showInlineCompletions: config.get('showInlineCompletions', true),
                semanticSearchEnabled: config.get('semanticSearch.enabled', true),
                semanticSearchMaxResults: config.get('semanticSearch.maxResults', 10),
                serverPort: config.get('serverPort', 8000),
                serverApiKey: config.get('serverApiKey', ''),
            },
        });
    }

    /**
     * Get display name for a model
     */
    private getModelDisplayName(provider: ProviderType, modelId: string): string {
        const modelNames: Record<string, string> = {
            'claude-sonnet-4-20250514': 'Claude Sonnet 4',
            'claude-opus-4-5-20251101': 'Claude Opus 4.5',
            'gpt-4-turbo': 'GPT-4 Turbo',
            'gpt-4o': 'GPT-4o',
            'gemini-2.0-flash': 'Gemini 2.0 Flash',
        };
        return modelNames[modelId] || modelId;
    }

    /**
     * Get the current state
     */
    getState(): Readonly<AppState> {
        return this.state;
    }

    /**
     * Get a specific part of the state using a selector
     */
    select<T>(selector: StateSelector<T>): T {
        return selector(this.state);
    }

    /**
     * Update the state (partial update with deep merge)
     */
    updateState(update: DeepPartial<AppState>): void {
        const previousState = { ...this.state };
        this.state = deepMerge(this.state, update);

        // Determine which top-level keys changed
        const changedKeys = Object.keys(update) as (keyof AppState)[];

        for (const key of changedKeys) {
            const event: StateChangeEvent = {
                key,
                previousValue: previousState[key],
                newValue: this.state[key],
                timestamp: Date.now(),
            };

            // Notify key-specific subscribers
            const keySubscribers = this.subscribers.get(key) || [];
            for (const subscriber of keySubscribers) {
                subscriber(this.state, event);
            }

            // Emit event
            this._onStateChange.fire(event);
        }

        // Notify selector subscribers
        this.notifySelectorSubscribers(previousState);

        // Persist state
        this.persistState();
    }

    /**
     * Notify selector-based subscribers
     */
    private notifySelectorSubscribers(previousState: AppState): void {
        for (const [selector, subscribers] of this.selectorSubscribers) {
            const previousValue = selector(previousState);
            const newValue = selector(this.state);

            // Only notify if the selected value changed
            if (previousValue !== newValue) {
                for (const subscriber of subscribers) {
                    subscriber(newValue);
                }
            }
        }
    }

    /**
     * Subscribe to state changes for a specific key
     */
    subscribe<K extends keyof AppState>(key: K, subscriber: StateSubscriber): vscode.Disposable {
        const subscribers = this.subscribers.get(key) || [];
        subscribers.push(subscriber);
        this.subscribers.set(key, subscribers);

        return {
            dispose: () => {
                const subs = this.subscribers.get(key) || [];
                const index = subs.indexOf(subscriber);
                if (index > -1) {
                    subs.splice(index, 1);
                }
            }
        };
    }

    /**
     * Subscribe to changes in a selected part of state
     */
    subscribeToSelector<T>(selector: StateSelector<T>, subscriber: StateSubscriber<T>): vscode.Disposable {
        const subscribers = this.selectorSubscribers.get(selector as StateSelector<unknown>) || new Set();
        subscribers.add(subscriber as StateSubscriber<unknown>);
        this.selectorSubscribers.set(selector as StateSelector<unknown>, subscribers);

        // Immediately call with current value
        subscriber(selector(this.state));

        return {
            dispose: () => {
                const subs = this.selectorSubscribers.get(selector as StateSelector<unknown>);
                if (subs) {
                    subs.delete(subscriber as StateSubscriber<unknown>);
                }
            }
        };
    }

    /**
     * Persist state to VS Code globalState
     */
    private async persistState(): Promise<void> {
        if (!this.globalState) return;

        // Only persist certain parts of state (not transient data)
        const stateToPersist: Partial<AppState> = {
            session: {
                mode: this.state.session.mode,
                model: this.state.session.model,
                conversation: this.state.session.conversation,
                context: [], // Don't persist context
            },
            // Persist non-sensitive settings only (avoid storing API key in VS Code storage)
            settings: {
                autoStart: this.state.settings.autoStart,
                showInlineCompletions: this.state.settings.showInlineCompletions,
                semanticSearchEnabled: this.state.settings.semanticSearchEnabled,
                semanticSearchMaxResults: this.state.settings.semanticSearchMaxResults,
                serverPort: this.state.settings.serverPort,
            },
        };

        await this.globalState.update(this.STORAGE_KEY, stateToPersist);
    }

    // ==================== Convenience Methods ====================

    /**
     * Set server status
     */
    setServerStatus(status: ServerStatus): void {
        this.updateState({
            server: { status }
        });
    }

    /**
     * Set connection state
     */
    setConnectionState(connectionState: ConnectionState, error?: string): void {
        this.updateState({
            server: {
                connectionState,
                lastError: error,
                lastHealthCheck: Date.now(),
            }
        });
    }

    /**
     * Set agent mode
     */
    async setMode(mode: AgentMode): Promise<void> {
        this.updateState({
            session: { mode },
            ui: { statusBarMode: mode },
        });

        // Also update VS Code configuration
        const config = vscode.workspace.getConfiguration('victor');
        await config.update('mode', mode, true);
    }

    /**
     * Set model
     */
    async setModel(model: ModelInfo): Promise<void> {
        this.updateState({
            session: { model }
        });

        // Also update VS Code configuration
        const config = vscode.workspace.getConfiguration('victor');
        await config.update('model', model.modelId, true);
        await config.update('provider', model.provider, true);
    }

    /**
     * Start new conversation
     */
    startNewConversation(): void {
        this.updateState({
            session: {
                conversation: {
                    id: generateId(),
                    messages: [],
                    isStreaming: false,
                    lastActivity: Date.now(),
                },
                context: [],
            }
        });
    }

    /**
     * Set streaming state
     */
    setStreaming(isStreaming: boolean): void {
        this.updateState({
            session: {
                conversation: {
                    isStreaming,
                    lastActivity: Date.now(),
                }
            }
        });
    }

    /**
     * Add context item
     */
    addContext(item: ContextItem): void {
        const currentContext = [...this.state.session.context];
        // Avoid duplicates
        const exists = currentContext.some(
            c => c.type === item.type && c.name === item.name
        );
        if (!exists) {
            currentContext.push(item);
            this.updateState({
                session: { context: currentContext }
            });
        }
    }

    /**
     * Remove context item
     */
    removeContext(name: string): void {
        const currentContext = this.state.session.context.filter(
            c => c.name !== name
        );
        this.updateState({
            session: { context: currentContext }
        });
    }

    /**
     * Clear all context
     */
    clearContext(): void {
        this.updateState({
            session: { context: [] }
        });
    }

    /**
     * Dispose the store
     */
    dispose(): void {
        this._onStateChange.dispose();
        this.subscribers.clear();
        this.selectorSubscribers.clear();
    }
}

// Export singleton accessor
export function getStore(): StateStore {
    return StateStore.getInstance();
}

// Export selectors for common state access patterns
export const selectors = {
    serverStatus: (state: AppState) => state.server.status,
    connectionState: (state: AppState) => state.server.connectionState,
    serverUrl: (state: AppState) => state.server.url,
    serverApiKey: (state: AppState) => state.settings.serverApiKey,
    mode: (state: AppState) => state.session.mode,
    model: (state: AppState) => state.session.model,
    conversation: (state: AppState) => state.session.conversation,
    context: (state: AppState) => state.session.context,
    isStreaming: (state: AppState) => state.session.conversation.isStreaming,
    settings: (state: AppState) => state.settings,
    isInitialized: (state: AppState) => state.initialized,
};
