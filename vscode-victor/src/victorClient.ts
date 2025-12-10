/**
 * Victor API Client
 *
 * Handles communication with the Victor backend server.
 * Features:
 * - HTTP/REST API calls with proper error handling
 * - WebSocket with auto-reconnection and heartbeat
 * - Centralized error categorization
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// =============================================================================
// Types and Interfaces
// =============================================================================

export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    toolCalls?: ToolCall[];
}

export interface ToolCall {
    id?: string;
    name: string;
    arguments: Record<string, unknown>;
    result?: unknown;
    // Enhanced fields for UI
    category?: string;
    cost_tier?: 'free' | 'low' | 'medium' | 'high';
    is_dangerous?: boolean;
    status?: 'pending' | 'running' | 'success' | 'error';
}

export interface CompletionRequest {
    prompt: string;
    file: string;
    language: string;
    position: { line: number; character: number };
    context?: string;
}

export interface SearchResult {
    file: string;
    line: number;
    content: string;
    score: number;
    context?: string;
}

export interface UndoRedoResult {
    success: boolean;
    message: string;
    files?: string[];
}

export interface ServerStatus {
    provider: string;
    model: string;
    mode: string;
    connected: boolean;
    capabilities?: string[];
}

// =============================================================================
// Error Handling
// =============================================================================

export enum VictorErrorType {
    Network = 'NETWORK',
    Timeout = 'TIMEOUT',
    ServerError = 'SERVER_ERROR',
    NotFound = 'NOT_FOUND',
    Validation = 'VALIDATION',
    Auth = 'AUTH',
    Unknown = 'UNKNOWN',
}

export class VictorError extends Error {
    constructor(
        message: string,
        public readonly type: VictorErrorType,
        public readonly statusCode?: number,
        public readonly originalError?: unknown
    ) {
        super(message);
        this.name = 'VictorError';
    }

    static fromAxiosError(error: AxiosError): VictorError {
        if (error.code === 'ECONNABORTED') {
            return new VictorError('Request timed out', VictorErrorType.Timeout, undefined, error);
        }
        if (error.code === 'ERR_NETWORK' || !error.response) {
            return new VictorError('Network error - server may be unavailable', VictorErrorType.Network, undefined, error);
        }

        const status = error.response.status;
        const message = (error.response.data as { message?: string })?.message || error.message;

        switch (status) {
            case 401:
            case 403:
                return new VictorError(message || 'Authentication failed', VictorErrorType.Auth, status, error);
            case 404:
                return new VictorError(message || 'Resource not found', VictorErrorType.NotFound, status, error);
            case 422:
                return new VictorError(message || 'Validation error', VictorErrorType.Validation, status, error);
            case 500:
            case 502:
            case 503:
                return new VictorError(message || 'Server error', VictorErrorType.ServerError, status, error);
            default:
                return new VictorError(message || 'Unknown error', VictorErrorType.Unknown, status, error);
        }
    }
}

// =============================================================================
// WebSocket Connection State
// =============================================================================

export enum WebSocketState {
    Disconnected = 'DISCONNECTED',
    Connecting = 'CONNECTING',
    Connected = 'CONNECTED',
    Reconnecting = 'RECONNECTING',
}

interface WebSocketConfig {
    reconnectAttempts: number;
    reconnectBaseDelay: number;
    reconnectMaxDelay: number;
    heartbeatInterval: number;
    heartbeatTimeout: number;
}

const DEFAULT_WS_CONFIG: WebSocketConfig = {
    reconnectAttempts: 5,
    reconnectBaseDelay: 1000,
    reconnectMaxDelay: 30000,
    heartbeatInterval: 30000,
    heartbeatTimeout: 10000,
};

// =============================================================================
// Victor Client
// =============================================================================

export class VictorClient {
    private client: AxiosInstance;
    private serverUrl: string;
    private apiToken?: string;
    private sessionToken?: string;

    // WebSocket state
    private wsConnection: WebSocket | null = null;
    private wsState: WebSocketState = WebSocketState.Disconnected;
    private wsConfig: WebSocketConfig;
    private wsReconnectAttempt = 0;
    private wsReconnectTimer: NodeJS.Timeout | null = null;
    private wsHeartbeatTimer: NodeJS.Timeout | null = null;
    private wsPongTimer: NodeJS.Timeout | null = null;

    // Event handlers
    private messageHandlers: ((msg: ChatMessage) => void)[] = [];
    private stateChangeHandlers: ((state: WebSocketState) => void)[] = [];
    private errorHandlers: ((error: VictorError) => void)[] = [];

    constructor(serverUrl: string, wsConfig?: Partial<WebSocketConfig>, apiToken?: string) {
        this.serverUrl = serverUrl;
        this.wsConfig = { ...DEFAULT_WS_CONFIG, ...wsConfig };
        this.apiToken = apiToken;

        this.client = axios.create({
            baseURL: serverUrl,
            timeout: 60000,
            headers: {
                'Content-Type': 'application/json',
                ...(apiToken ? { Authorization: `Bearer ${apiToken}` } : {}),
            },
        });
    }

    // =========================================================================
    // Connection Management
    // =========================================================================

    setApiToken(token?: string): void {
        this.apiToken = token;
        if (token) {
            this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        } else {
            delete this.client.defaults.headers.common['Authorization'];
        }
    }

    /**
     * Prefetch a session token over HTTP (auth required).
     * Useful for obtaining a token before opening the WebSocket.
     */
    async prefetchSessionToken(): Promise<string | undefined> {
        if (!this.apiToken) {
            return undefined;
        }
        try {
            const resp = await this.client.post('/session/token', {});
            const token = (resp.data as { session_token?: string }).session_token;
            if (typeof token === 'string') {
                this.sessionToken = token;
                return token;
            }
            return undefined;
        } catch (error) {
            // Surface as VictorError for listeners but don't throw hard
            try {
                const ve = VictorError.fromAxiosError(error as AxiosError);
                this.emitError(ve);
            } catch {
                // ignore conversion errors
            }
            return undefined;
        }
    }

    async checkConnection(): Promise<boolean> {
        try {
            const response = await this.client.get('/health');
            return response.status === 200;
        } catch {
            return false;
        }
    }

    getWebSocketState(): WebSocketState {
        return this.wsState;
    }

    // =========================================================================
    // Event Handlers
    // =========================================================================

    onMessage(handler: (msg: ChatMessage) => void): () => void {
        this.messageHandlers.push(handler);
        return () => {
            const index = this.messageHandlers.indexOf(handler);
            if (index !== -1) {
                this.messageHandlers.splice(index, 1);
            }
        };
    }

    onStateChange(handler: (state: WebSocketState) => void): () => void {
        this.stateChangeHandlers.push(handler);
        return () => {
            const index = this.stateChangeHandlers.indexOf(handler);
            if (index !== -1) {
                this.stateChangeHandlers.splice(index, 1);
            }
        };
    }

    onError(handler: (error: VictorError) => void): () => void {
        this.errorHandlers.push(handler);
        return () => {
            const index = this.errorHandlers.indexOf(handler);
            if (index !== -1) {
                this.errorHandlers.splice(index, 1);
            }
        };
    }

    private emitStateChange(state: WebSocketState): void {
        this.wsState = state;
        for (const handler of this.stateChangeHandlers) {
            try {
                handler(state);
            } catch (e) {
                console.error('State change handler error:', e);
            }
        }
    }

    private emitError(error: VictorError): void {
        for (const handler of this.errorHandlers) {
            try {
                handler(error);
            } catch (e) {
                console.error('Error handler error:', e);
            }
        }
    }

    // =========================================================================
    // WebSocket with Auto-Reconnection
    // =========================================================================

    private buildWebSocketUrl(): string {
        const base = this.serverUrl.replace(/^http/, 'ws') + '/ws';
        const url = new URL(base);
        if (this.apiToken) {
            url.searchParams.set('api_key', this.apiToken);
        }
        if (this.sessionToken) {
            url.searchParams.set('session_token', this.sessionToken);
        }
        return url.toString();
    }

    connectWebSocket(): void {
        if (this.wsState === WebSocketState.Connected || this.wsState === WebSocketState.Connecting) {
            return;
        }

        this.emitStateChange(WebSocketState.Connecting);
        this.wsReconnectAttempt = 0;
        this._connect();
    }

    private _connect(): void {
        try {
            const wsUrl = this.buildWebSocketUrl();
            this.wsConnection = new WebSocket(wsUrl);

            this.wsConnection.onopen = () => {
                console.log('Victor WebSocket connected');
                this.wsReconnectAttempt = 0;
                this.emitStateChange(WebSocketState.Connected);
                this._startHeartbeat();
            };

            this.wsConnection.onmessage = (event) => {
                try {
                    // Handle legacy string control messages
                    if (typeof event.data === 'string' && event.data.startsWith('[session]')) {
                        this.sessionToken = event.data.replace('[session]', '').trim();
                        return;
                    }

                    const data = JSON.parse(event.data);

                    // Handle pong response
                    if (data.type === 'pong') {
                        this._handlePong();
                        return;
                    }

                    if (data.type === 'session' && typeof data.token === 'string') {
                        this.sessionToken = data.token;
                        return;
                    }

                    // Handle chat messages
                    for (const handler of this.messageHandlers) {
                        try {
                            handler(data);
                        } catch (e) {
                            console.error('Message handler error:', e);
                        }
                    }
                } catch {
                    // Ignore parse errors
                }
            };

            this.wsConnection.onclose = (event) => {
                console.log('Victor WebSocket closed:', event.code, event.reason);
                this._stopHeartbeat();
                this.wsConnection = null;

                // Only reconnect if not intentionally closed
                if (event.code !== 1000) {
                    this._scheduleReconnect();
                } else {
                    this.emitStateChange(WebSocketState.Disconnected);
                }
            };

            this.wsConnection.onerror = (event) => {
                console.error('Victor WebSocket error:', event);
                this.emitError(new VictorError(
                    'WebSocket connection error',
                    VictorErrorType.Network
                ));
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this._scheduleReconnect();
        }
    }

    private _scheduleReconnect(): void {
        if (this.wsReconnectAttempt >= this.wsConfig.reconnectAttempts) {
            console.log('Max reconnection attempts reached');
            this.emitStateChange(WebSocketState.Disconnected);
            this.emitError(new VictorError(
                'Failed to reconnect after multiple attempts',
                VictorErrorType.Network
            ));
            return;
        }

        this.emitStateChange(WebSocketState.Reconnecting);

        // Exponential backoff
        const delay = Math.min(
            this.wsConfig.reconnectBaseDelay * Math.pow(2, this.wsReconnectAttempt),
            this.wsConfig.reconnectMaxDelay
        );

        console.log(`Reconnecting in ${delay}ms (attempt ${this.wsReconnectAttempt + 1}/${this.wsConfig.reconnectAttempts})`);

        this.wsReconnectTimer = setTimeout(() => {
            this.wsReconnectAttempt++;
            this._connect();
        }, delay);
    }

    private _startHeartbeat(): void {
        this._stopHeartbeat();

        this.wsHeartbeatTimer = setInterval(() => {
            if (this.wsConnection?.readyState === WebSocket.OPEN) {
                this.wsConnection.send(JSON.stringify({ type: 'ping' }));

                // Clear any existing pong timer to prevent race conditions
                if (this.wsPongTimer) {
                    clearTimeout(this.wsPongTimer);
                }

                // Set timeout for pong response
                this.wsPongTimer = setTimeout(() => {
                    console.log('Heartbeat timeout - closing connection');
                    this.wsConnection?.close(4000, 'Heartbeat timeout');
                }, this.wsConfig.heartbeatTimeout);
            }
        }, this.wsConfig.heartbeatInterval);
    }

    private _handlePong(): void {
        if (this.wsPongTimer) {
            clearTimeout(this.wsPongTimer);
            this.wsPongTimer = null;
        }
    }

    private _stopHeartbeat(): void {
        if (this.wsHeartbeatTimer) {
            clearInterval(this.wsHeartbeatTimer);
            this.wsHeartbeatTimer = null;
        }
        if (this.wsPongTimer) {
            clearTimeout(this.wsPongTimer);
            this.wsPongTimer = null;
        }
    }

    disconnectWebSocket(): void {
        this._stopHeartbeat();

        if (this.wsReconnectTimer) {
            clearTimeout(this.wsReconnectTimer);
            this.wsReconnectTimer = null;
        }

        if (this.wsConnection) {
            this.wsConnection.close(1000, 'Client disconnect');
            this.wsConnection = null;
        }

        this.emitStateChange(WebSocketState.Disconnected);
    }

    // =========================================================================
    // Chat API
    // =========================================================================

    async chat(messages: ChatMessage[]): Promise<ChatMessage> {
        try {
            const response = await this.client.post('/chat', { messages });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async streamChat(
        messages: ChatMessage[],
        onChunk: (chunk: string) => void,
        onToolCall?: (toolCall: ToolCall) => void
    ): Promise<void> {
        try {
            const response = await this.client.post('/chat/stream', { messages }, {
                responseType: 'stream',
            });

            return new Promise((resolve, reject) => {
                let buffer = '';

                response.data.on('data', (chunk: Buffer) => {
                    buffer += chunk.toString();
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.type === 'content') {
                                    onChunk(data.content);
                                } else if (data.type === 'tool_call' && onToolCall) {
                                    onToolCall(data.tool_call);
                                } else if (data.type === 'error') {
                                    reject(new VictorError(data.message, VictorErrorType.ServerError));
                                }
                            } catch {
                                // Ignore parse errors for incomplete chunks
                            }
                        }
                    }
                });

                response.data.on('end', () => resolve());
                response.data.on('error', (err: Error) => reject(this._handleError(err)));
            });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // =========================================================================
    // Completions API
    // =========================================================================

    async getCompletions(request: CompletionRequest): Promise<string[]> {
        try {
            const response = await this.client.post('/completions', request);
            return response.data.completions || [];
        } catch (error) {
            console.error('Completions error:', error);
            return []; // Graceful degradation for completions
        }
    }

    // =========================================================================
    // Search API
    // =========================================================================

    async semanticSearch(query: string, maxResults: number = 10): Promise<SearchResult[]> {
        try {
            const response = await this.client.post('/search/semantic', {
                query,
                max_results: maxResults,
            });
            return response.data.results || [];
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async codeSearch(query: string, options?: {
        regex?: boolean;
        caseSensitive?: boolean;
        filePattern?: string;
    }): Promise<SearchResult[]> {
        try {
            const response = await this.client.post('/search/code', {
                query,
                ...options,
            });
            return response.data.results || [];
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // =========================================================================
    // Model/Mode API
    // =========================================================================

    async switchModel(provider: string, model: string): Promise<void> {
        try {
            await this.client.post('/model/switch', { provider, model });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async switchMode(mode: string): Promise<void> {
        try {
            await this.client.post('/mode/switch', { mode });
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async getStatus(): Promise<ServerStatus> {
        try {
            const response = await this.client.get('/status');
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Get list of available models from all providers.
     */
    async getModels(): Promise<{
        provider: string;
        model_id: string;
        display_name: string;
        is_local: boolean;
    }[]> {
        try {
            const response = await this.client.get('/models');
            return response.data.models || [];
        } catch (error) {
            console.error('Get models error:', error);
            return []; // Graceful degradation
        }
    }

    /**
     * Get list of available LLM providers with their configuration status.
     */
    async getProviders(): Promise<{
        name: string;
        display_name: string;
        is_local: boolean;
        configured: boolean;
        supports_tools: boolean;
        supports_streaming: boolean;
    }[]> {
        try {
            const response = await this.client.get('/providers');
            return response.data.providers || [];
        } catch (error) {
            console.error('Get providers error:', error);
            return []; // Graceful degradation
        }
    }

    /**
     * Get list of available tools with their metadata.
     */
    async getTools(): Promise<{
        tools: {
            name: string;
            description: string;
            category: string;
            cost_tier: string;
            parameters: Record<string, unknown>;
            is_dangerous: boolean;
            requires_approval: boolean;
        }[];
        total: number;
        categories: string[];
    }> {
        try {
            const response = await this.client.get('/tools');
            return {
                tools: response.data.tools || [],
                total: response.data.total || 0,
                categories: response.data.categories || [],
            };
        } catch (error) {
            console.error('Get tools error:', error);
            return { tools: [], total: 0, categories: [] }; // Graceful degradation
        }
    }

    // =========================================================================
    // Undo/Redo API
    // =========================================================================

    async undo(): Promise<UndoRedoResult> {
        try {
            const response = await this.client.post('/undo');
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async redo(): Promise<UndoRedoResult> {
        try {
            const response = await this.client.post('/redo');
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // =========================================================================
    // Conversation API
    // =========================================================================

    async resetConversation(): Promise<void> {
        try {
            await this.client.post('/conversation/reset');
        } catch (error) {
            throw this._handleError(error);
        }
    }

    async exportConversation(format: 'json' | 'markdown' = 'json'): Promise<string> {
        try {
            const response = await this.client.get('/conversation/export', {
                params: { format },
            });
            return format === 'json' ? JSON.stringify(response.data) : response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // =========================================================================
    // History API
    // =========================================================================

    async getHistory(limit: number = 10): Promise<{
        id: string;
        timestamp: string;
        toolName: string;
        description: string;
        fileCount: number;
    }[]> {
        try {
            const response = await this.client.get('/history', {
                params: { limit },
            });
            return response.data.history || [];
        } catch (error) {
            console.error('History error:', error);
            return []; // Graceful degradation
        }
    }

    // =========================================================================
    // Patch API
    // =========================================================================

    async applyPatch(patch: string, dryRun: boolean = false): Promise<{
        success: boolean;
        filesModified: string[];
        preview?: string;
    }> {
        try {
            const response = await this.client.post('/patch/apply', {
                patch,
                dry_run: dryRun,
            });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // =========================================================================
    // LSP API
    // =========================================================================

    async getDefinition(file: string, line: number, character: number): Promise<{
        file: string;
        line: number;
        character: number;
    }[]> {
        try {
            const response = await this.client.post('/lsp/definition', {
                file,
                line,
                character,
            });
            return response.data.locations || [];
        } catch (error) {
            console.error('LSP definition error:', error);
            return []; // Graceful degradation
        }
    }

    async getReferences(file: string, line: number, character: number): Promise<{
        file: string;
        line: number;
        character: number;
    }[]> {
        try {
            const response = await this.client.post('/lsp/references', {
                file,
                line,
                character,
            });
            return response.data.locations || [];
        } catch (error) {
            console.error('LSP references error:', error);
            return []; // Graceful degradation
        }
    }

    async getHover(file: string, line: number, character: number): Promise<string | null> {
        try {
            const response = await this.client.post('/lsp/hover', {
                file,
                line,
                character,
            });
            return response.data.contents || null;
        } catch {
            return null; // Graceful degradation
        }
    }

    async getDiagnostics(file: string): Promise<{
        line: number;
        character: number;
        message: string;
        severity: string;
    }[]> {
        try {
            const response = await this.client.post('/lsp/diagnostics', { file });
            return response.data.diagnostics || [];
        } catch (error) {
            console.error('LSP diagnostics error:', error);
            return []; // Graceful degradation
        }
    }

    // =========================================================================
    // Server Control
    // =========================================================================

    async stopServer(): Promise<void> {
        try {
            await this.client.post('/shutdown');
        } catch (error) {
            // Ignore errors - server might close before response
        }
    }

    // =========================================================================
    // Workspace Analysis API
    // =========================================================================

    /**
     * Get workspace structure overview.
     */
    async getWorkspaceOverview(depth: number = 3): Promise<{
        root: string;
        name: string;
        tree: unknown;
        file_counts: Record<string, number>;
        total_files: number;
        total_size: number;
    }> {
        try {
            const response = await this.client.get('/workspace/overview', {
                params: { depth },
            });
            return response.data;
        } catch (error) {
            console.error('Workspace overview error:', error);
            return {
                root: '',
                name: '',
                tree: null,
                file_counts: {},
                total_files: 0,
                total_size: 0,
            };
        }
    }

    /**
     * Get code metrics for the workspace.
     */
    async getWorkspaceMetrics(): Promise<{
        lines_of_code: number;
        files_by_type: Record<string, number>;
        largest_files: { path: string; lines: number; size: number }[];
    }> {
        try {
            const response = await this.client.get('/workspace/metrics');
            return response.data;
        } catch (error) {
            console.error('Workspace metrics error:', error);
            return {
                lines_of_code: 0,
                files_by_type: {},
                largest_files: [],
            };
        }
    }

    /**
     * Get security scan results for the workspace.
     */
    async getWorkspaceSecurity(): Promise<{
        scan_completed: boolean;
        findings: {
            file: string;
            line: number;
            type: string;
            severity: string;
            snippet: string;
        }[];
        total_findings: number;
        severity_counts: Record<string, number>;
    }> {
        try {
            const response = await this.client.get('/workspace/security');
            return response.data;
        } catch (error) {
            console.error('Workspace security error:', error);
            return {
                scan_completed: false,
                findings: [],
                total_findings: 0,
                severity_counts: {},
            };
        }
    }

    /**
     * Get dependency information for the workspace.
     */
    async getWorkspaceDependencies(): Promise<{
        workspace: string;
        dependencies: Record<string, {
            file: string;
            count?: number;
            packages?: string[];
        }>;
    }> {
        try {
            const response = await this.client.get('/workspace/dependencies');
            return response.data;
        } catch (error) {
            console.error('Workspace dependencies error:', error);
            return {
                workspace: '',
                dependencies: {},
            };
        }
    }

    // =========================================================================
    // Tool Approval API
    // =========================================================================

    /**
     * Approve or reject a pending tool execution.
     */
    async approveTool(approvalId: string, approved: boolean): Promise<{
        success: boolean;
        approval_id: string;
        approved: boolean;
    }> {
        try {
            const response = await this.client.post('/tools/approve', {
                approval_id: approvalId,
                approved,
            });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Get list of pending tool approvals.
     */
    async getPendingApprovals(): Promise<{
        pending: {
            approval_id: string;
            tool_name: string;
            arguments: Record<string, unknown>;
            danger_level: string;
            cost_tier: string;
            created_at: string;
        }[];
        count: number;
    }> {
        try {
            const response = await this.client.get('/tools/pending');
            return response.data;
        } catch (error) {
            console.error('Pending approvals error:', error);
            return { pending: [], count: 0 };
        }
    }

    // =========================================================================
    // API Key Management API (System Keyring)
    // =========================================================================

    /**
     * Store API key in system keyring (macOS Keychain, Windows Credential Manager, etc.)
     * via the Victor backend which uses Python keyring library.
     */
    async setApiKey(providerName: string, apiKey: string): Promise<{ success: boolean }> {
        try {
            const response = await this.client.post('/credentials/set', {
                provider: providerName,
                api_key: apiKey,
            });
            return { success: response.data.success || true };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Retrieve API key from system keyring via Victor backend.
     */
    async getApiKey(providerName: string): Promise<{ apiKey: string | null }> {
        try {
            const response = await this.client.get('/credentials/get', {
                params: { provider: providerName },
            });
            return { apiKey: response.data.api_key || null };
        } catch (error) {
            // Not found or unavailable
            return { apiKey: null };
        }
    }

    /**
     * Delete API key from system keyring via Victor backend.
     */
    async deleteApiKey(providerName: string): Promise<{ success: boolean }> {
        try {
            const response = await this.client.delete('/credentials/delete', {
                params: { provider: providerName },
            });
            return { success: response.data.success || true };
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Check if system keyring is available.
     */
    async checkKeyringAvailable(): Promise<{ available: boolean; backend: string }> {
        try {
            const response = await this.client.get('/credentials/status');
            return {
                available: response.data.available || false,
                backend: response.data.backend || 'unknown',
            };
        } catch (error) {
            return { available: false, backend: 'unavailable' };
        }
    }

    // =========================================================================
    // Git Integration API
    // =========================================================================

    /**
     * Get git status for the workspace.
     */
    async getGitStatus(): Promise<{
        is_git_repo: boolean;
        branch?: string;
        tracking?: string;
        staged?: { status: string; file: string }[];
        unstaged?: { status: string; file: string }[];
        untracked?: string[];
        is_clean?: boolean;
        error?: string;
    }> {
        try {
            const response = await this.client.get('/git/status');
            return response.data;
        } catch (error) {
            console.error('Git status error:', error);
            return { is_git_repo: false, error: 'Failed to get git status' };
        }
    }

    /**
     * Create a git commit with optional AI-generated message.
     */
    async gitCommit(options: {
        message?: string;
        use_ai?: boolean;
        files?: string[];
    }): Promise<{
        success: boolean;
        message?: string;
        output?: string;
        error?: string;
    }> {
        try {
            const response = await this.client.post('/git/commit', options);
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Get git commit log.
     */
    async getGitLog(limit: number = 20): Promise<{
        commits: {
            hash: string;
            author: string;
            email: string;
            relative_date: string;
            message: string;
        }[];
    }> {
        try {
            const response = await this.client.get('/git/log', {
                params: { limit },
            });
            return response.data;
        } catch (error) {
            console.error('Git log error:', error);
            return { commits: [] };
        }
    }

    /**
     * Get git diff.
     */
    async getGitDiff(options?: {
        staged?: boolean;
        file?: string;
    }): Promise<{
        diff: string;
        truncated: boolean;
    }> {
        try {
            const response = await this.client.get('/git/diff', {
                params: options,
            });
            return response.data;
        } catch (error) {
            console.error('Git diff error:', error);
            return { diff: '', truncated: false };
        }
    }

    // =========================================================================
    // MCP Integration API
    // =========================================================================

    /**
     * Get list of configured MCP servers.
     */
    async getMcpServers(): Promise<{
        servers: {
            name: string;
            connected: boolean;
            tools: string[];
            endpoint?: string;
        }[];
    }> {
        try {
            const response = await this.client.get('/mcp/servers');
            return response.data;
        } catch (error) {
            console.error('MCP servers error:', error);
            return { servers: [] };
        }
    }

    /**
     * Connect to an MCP server.
     */
    async connectMcpServer(server: string, endpoint?: string): Promise<{
        success: boolean;
        server: string;
    }> {
        try {
            const response = await this.client.post('/mcp/connect', {
                server,
                endpoint,
            });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Disconnect from an MCP server.
     */
    async disconnectMcpServer(server: string): Promise<{
        success: boolean;
        server: string;
    }> {
        try {
            const response = await this.client.post('/mcp/disconnect', { server });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // =========================================================================
    // RL Model Selector API
    // =========================================================================

    /**
     * Get RL model selector statistics.
     */
    async getRLStats(): Promise<{
        total_selections: number;
        epsilon: number;
        strategy: string;
        q_table: Record<string, number>;
        selection_counts: Record<string, number>;
        q_table_by_task?: Record<string, Record<string, number>>;
    }> {
        try {
            const response = await this.client.get('/rl/stats');
            return response.data;
        } catch (error) {
            console.error('RL stats error:', error);
            return {
                total_selections: 0,
                epsilon: 0.1,
                strategy: 'epsilon_greedy',
                q_table: {},
                selection_counts: {},
            };
        }
    }

    /**
     * Get model recommendation from RL selector.
     */
    async getRLRecommendation(taskType?: string): Promise<{
        recommended: string;
        strategy: string;
        exploration_rate: number;
        q_values: Record<string, number>;
        available_providers: string[];
    }> {
        try {
            const params = taskType ? { task_type: taskType } : {};
            const response = await this.client.get('/rl/recommend', { params });
            return response.data;
        } catch (error) {
            console.error('RL recommend error:', error);
            return {
                recommended: 'ollama',
                strategy: 'epsilon_greedy',
                exploration_rate: 0.1,
                q_values: {},
                available_providers: [],
            };
        }
    }

    /**
     * Set RL exploration rate.
     */
    async setRLExplorationRate(rate: number): Promise<{
        success: boolean;
        previous_rate: number;
        new_rate: number;
    }> {
        try {
            const response = await this.client.post('/rl/explore', { rate });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Set RL selection strategy.
     */
    async setRLStrategy(strategy: string): Promise<{
        success: boolean;
        previous_strategy: string;
        new_strategy: string;
    }> {
        try {
            const response = await this.client.post('/rl/strategy', { strategy });
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    /**
     * Reset RL Q-values.
     */
    async resetRLQValues(): Promise<{
        success: boolean;
        message: string;
    }> {
        try {
            const response = await this.client.post('/rl/reset');
            return response.data;
        } catch (error) {
            throw this._handleError(error);
        }
    }

    // =========================================================================
    // Error Handling
    // =========================================================================

    private _handleError(error: unknown): VictorError {
        if (error instanceof VictorError) {
            return error;
        }
        if (axios.isAxiosError(error)) {
            return VictorError.fromAxiosError(error);
        }
        if (error instanceof Error) {
            return new VictorError(error.message, VictorErrorType.Unknown, undefined, error);
        }
        return new VictorError('Unknown error occurred', VictorErrorType.Unknown, undefined, error);
    }
}
