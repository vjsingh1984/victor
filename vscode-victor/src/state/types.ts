/**
 * State Types
 *
 * Type definitions for the centralized state store.
 */

import * as vscode from 'vscode';
import { ServerStatus } from '../serverManager';

/**
 * Agent modes available in Victor
 */
export type AgentMode = 'build' | 'plan' | 'explore';

/**
 * LLM Provider types
 */
export type ProviderType = 'anthropic' | 'openai' | 'ollama' | 'google' | 'lmstudio' | 'vllm' | 'xai';

/**
 * Connection state for server/WebSocket
 */
export enum ConnectionState {
    Disconnected = 'disconnected',
    Connecting = 'connecting',
    Connected = 'connected',
    Reconnecting = 'reconnecting',
    Error = 'error',
}

/**
 * Model information
 */
export interface ModelInfo {
    provider: ProviderType;
    modelId: string;
    displayName: string;
    capabilities?: ModelCapabilities;
}

/**
 * Model capabilities
 */
export interface ModelCapabilities {
    streaming: boolean;
    tools: boolean;
    vision: boolean;
    maxTokens: number;
}

/**
 * Conversation message
 */
export interface ConversationMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: number;
    toolCalls?: ToolCallInfo[];
}

/**
 * Tool call information
 */
export interface ToolCallInfo {
    id: string;
    name: string;
    arguments: Record<string, unknown>;
    result?: string;
    status: 'pending' | 'running' | 'completed' | 'error';
}

/**
 * Conversation state
 */
export interface ConversationState {
    id: string;
    messages: ConversationMessage[];
    isStreaming: boolean;
    lastActivity: number;
}

/**
 * Server state
 */
export interface ServerState {
    status: ServerStatus;
    url: string;
    port: number;
    connectionState: ConnectionState;
    lastError?: string;
    lastHealthCheck?: number;
}

/**
 * Context item (for @-mentions)
 */
export interface ContextItem {
    type: 'file' | 'symbol' | 'selection' | 'folder' | 'git-diff' | 'terminal';
    uri?: vscode.Uri;
    name: string;
    content?: string;
    range?: vscode.Range;
}

/**
 * Session state
 */
export interface SessionState {
    mode: AgentMode;
    model: ModelInfo;
    conversation: ConversationState;
    context: ContextItem[];
}

/**
 * UI state
 */
export interface UIState {
    chatViewVisible: boolean;
    sidebarOpen: boolean;
    activePanel: string | null;
    statusBarMode: AgentMode;
}

/**
 * Settings state (synced with VS Code configuration)
 */
export interface SettingsState {
    autoStart: boolean;
    showInlineCompletions: boolean;
    semanticSearchEnabled: boolean;
    semanticSearchMaxResults: number;
    serverPort: number;
    serverApiKey?: string;
}

/**
 * Complete application state
 */
export interface AppState {
    server: ServerState;
    session: SessionState;
    ui: UIState;
    settings: SettingsState;
    initialized: boolean;
}

/**
 * State change event
 */
export interface StateChangeEvent<K extends keyof AppState = keyof AppState> {
    key: K;
    previousValue: AppState[K];
    newValue: AppState[K];
    timestamp: number;
}

/**
 * State subscriber callback
 */
export type StateSubscriber<T = AppState> = (state: T, event?: StateChangeEvent) => void;

/**
 * State selector function
 */
export type StateSelector<T> = (state: AppState) => T;

/**
 * Partial state update (deep partial)
 */
export type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};
