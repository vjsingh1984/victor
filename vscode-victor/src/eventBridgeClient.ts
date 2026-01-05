/**
 * EventBridge Client for VS Code
 *
 * Connects to Victor's EventBridge WebSocket endpoint for real-time
 * updates on tool execution, file changes, and other events.
 *
 * Architecture:
 *   Victor Server (EventBus) → EventBridge → WebSocket → This Client
 */

import * as vscode from 'vscode';
import WebSocket from 'ws';

export interface VictorEvent {
    id: string;
    type: string;
    data: Record<string, unknown>;
    timestamp: number;
}

export type EventHandler = (event: VictorEvent) => void;

export enum ConnectionState {
    Disconnected = 'disconnected',
    Connecting = 'connecting',
    Connected = 'connected',
    Reconnecting = 'reconnecting',
    Error = 'error'
}

interface ReconnectConfig {
    initialDelayMs: number;
    maxDelayMs: number;
    multiplier: number;
    maxRetries: number;
}

const DEFAULT_RECONNECT_CONFIG: ReconnectConfig = {
    initialDelayMs: 500,
    maxDelayMs: 30000,
    multiplier: 2,
    maxRetries: 10
};

/**
 * Client for connecting to Victor's EventBridge WebSocket endpoint.
 *
 * Provides:
 * - Automatic reconnection with exponential backoff
 * - Event filtering by type
 * - Connection state tracking
 * - Graceful cleanup
 */
export class EventBridgeClient {
    private ws: WebSocket | null = null;
    private serverUrl: string = '';
    private state: ConnectionState = ConnectionState.Disconnected;
    private reconnectAttempt: number = 0;
    private reconnectConfig: ReconnectConfig;
    private eventHandlers: Map<string, Set<EventHandler>> = new Map();
    private globalHandlers: Set<EventHandler> = new Set();
    private stateChangeHandlers: Set<(state: ConnectionState) => void> = new Set();
    private outputChannel: vscode.OutputChannel;
    private reconnectTimer: NodeJS.Timeout | null = null;
    private pingInterval: NodeJS.Timeout | null = null;

    constructor(reconnectConfig: ReconnectConfig = DEFAULT_RECONNECT_CONFIG) {
        this.reconnectConfig = reconnectConfig;
        this.outputChannel = vscode.window.createOutputChannel('Victor Events');
    }

    /**
     * Connect to the EventBridge WebSocket endpoint.
     *
     * @param serverUrl Base URL of the Victor server (e.g., http://127.0.0.1:8765)
     */
    connect(serverUrl: string): void {
        this.serverUrl = serverUrl;
        this.doConnect();
    }

    /**
     * Disconnect from the EventBridge.
     */
    disconnect(): void {
        this.stopReconnect();
        this.stopPing();

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.setState(ConnectionState.Disconnected);
    }

    /**
     * Get the current connection state.
     */
    getState(): ConnectionState {
        return this.state;
    }

    /**
     * Subscribe to specific event types.
     *
     * @param eventType Event type to listen for (e.g., "tool.start", "file.modified")
     * @param handler Callback function to handle the event
     * @returns Disposable to unsubscribe
     */
    on(eventType: string, handler: EventHandler): vscode.Disposable {
        if (!this.eventHandlers.has(eventType)) {
            this.eventHandlers.set(eventType, new Set());
        }
        this.eventHandlers.get(eventType)!.add(handler);

        return new vscode.Disposable(() => {
            this.eventHandlers.get(eventType)?.delete(handler);
        });
    }

    /**
     * Subscribe to all events.
     *
     * @param handler Callback function to handle all events
     * @returns Disposable to unsubscribe
     */
    onAny(handler: EventHandler): vscode.Disposable {
        this.globalHandlers.add(handler);
        return new vscode.Disposable(() => {
            this.globalHandlers.delete(handler);
        });
    }

    /**
     * Subscribe to connection state changes.
     *
     * @param handler Callback function for state changes
     * @returns Disposable to unsubscribe
     */
    onStateChange(handler: (state: ConnectionState) => void): vscode.Disposable {
        this.stateChangeHandlers.add(handler);
        return new vscode.Disposable(() => {
            this.stateChangeHandlers.delete(handler);
        });
    }

    /**
     * Show the output channel.
     */
    showOutput(): void {
        this.outputChannel.show();
    }

    /**
     * Dispose of resources.
     */
    dispose(): void {
        this.disconnect();
        this.eventHandlers.clear();
        this.globalHandlers.clear();
        this.stateChangeHandlers.clear();
        this.outputChannel.dispose();
    }

    // --- Private Methods ---

    private doConnect(): void {
        if (this.state === ConnectionState.Connected || this.state === ConnectionState.Connecting) {
            return;
        }

        this.setState(ConnectionState.Connecting);

        // Convert HTTP URL to WebSocket URL
        const wsUrl = this.serverUrl.replace(/^http/, 'ws') + '/ws/events';
        this.log(`Connecting to EventBridge: ${wsUrl}`);

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.on('open', () => {
                this.log('Connected to EventBridge');
                this.setState(ConnectionState.Connected);
                this.reconnectAttempt = 0;
                this.startPing();

                // Send subscription message
                this.ws?.send(JSON.stringify({
                    type: 'subscribe',
                    categories: ['all']
                }));
            });

            this.ws.on('message', (data: WebSocket.Data) => {
                try {
                    const message = JSON.parse(data.toString());
                    if (message.type === 'event') {
                        this.handleEvent(message.event);
                    } else if (message.type === 'pong') {
                        // Ping response received
                    }
                } catch (error) {
                    this.log(`Error parsing message: ${error}`);
                }
            });

            this.ws.on('close', (code, reason) => {
                this.log(`Disconnected from EventBridge (code: ${code}, reason: ${reason})`);
                this.ws = null;
                this.stopPing();

                if (this.state !== ConnectionState.Disconnected) {
                    this.scheduleReconnect();
                }
            });

            this.ws.on('error', (error) => {
                this.log(`WebSocket error: ${error.message}`);
                this.setState(ConnectionState.Error);
            });

        } catch (error) {
            this.log(`Failed to connect: ${error}`);
            this.setState(ConnectionState.Error);
            this.scheduleReconnect();
        }
    }

    private handleEvent(event: VictorEvent): void {
        this.log(`Event: ${event.type} - ${JSON.stringify(event.data)}`);

        // Call type-specific handlers
        const handlers = this.eventHandlers.get(event.type);
        if (handlers) {
            for (const handler of handlers) {
                try {
                    handler(event);
                } catch (error) {
                    this.log(`Error in event handler: ${error}`);
                }
            }
        }

        // Call global handlers
        for (const handler of this.globalHandlers) {
            try {
                handler(event);
            } catch (error) {
                this.log(`Error in global handler: ${error}`);
            }
        }
    }

    private scheduleReconnect(): void {
        if (this.reconnectAttempt >= this.reconnectConfig.maxRetries) {
            this.log(`Max reconnection attempts (${this.reconnectConfig.maxRetries}) reached`);
            this.setState(ConnectionState.Error);
            return;
        }

        this.reconnectAttempt++;
        const delay = Math.min(
            this.reconnectConfig.initialDelayMs * Math.pow(this.reconnectConfig.multiplier, this.reconnectAttempt - 1),
            this.reconnectConfig.maxDelayMs
        );

        this.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempt}/${this.reconnectConfig.maxRetries})`);
        this.setState(ConnectionState.Reconnecting);

        this.reconnectTimer = setTimeout(() => {
            this.doConnect();
        }, delay);
    }

    private stopReconnect(): void {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    private startPing(): void {
        this.stopPing();
        // Send ping every 30 seconds to keep connection alive
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    private stopPing(): void {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    private setState(state: ConnectionState): void {
        if (this.state !== state) {
            this.state = state;
            for (const handler of this.stateChangeHandlers) {
                try {
                    handler(state);
                } catch (error) {
                    this.log(`Error in state change handler: ${error}`);
                }
            }
        }
    }

    private log(message: string): void {
        const timestamp = new Date().toISOString();
        this.outputChannel.appendLine(`[${timestamp}] ${message}`);
    }
}

/**
 * Create an EventBridge client singleton.
 */
let eventBridgeInstance: EventBridgeClient | null = null;

export function getEventBridgeClient(): EventBridgeClient {
    if (!eventBridgeInstance) {
        eventBridgeInstance = new EventBridgeClient();
    }
    return eventBridgeInstance;
}

export function disposeEventBridgeClient(): void {
    if (eventBridgeInstance) {
        eventBridgeInstance.dispose();
        eventBridgeInstance = null;
    }
}
