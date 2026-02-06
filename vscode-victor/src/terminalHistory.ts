/**
 * Terminal History Service
 *
 * Shared service for tracking terminal command history across the extension.
 * Uses VS Code's Shell Integration API (when available) for command capture.
 * Used by both inline completions and terminal agent for context.
 *
 * @see https://code.visualstudio.com/docs/terminal/shell-integration
 */

import * as vscode from 'vscode';

export interface TerminalHistoryEntry {
    command: string;
    timestamp: number;
    terminalName: string;
    workingDir?: string;
    exitCode?: number;
    output?: string;
}

// Shell execution event interfaces (VS Code API)
interface ShellExecutionStartEvent {
    terminal: vscode.Terminal;
    shellIntegration?: {
        executeCommand: {
            commandLine?: { value: string };
            cwd?: { fsPath: string };
            read?: () => vscode.ReadableStream<string>;
        };
    };
    execution?: {
        commandLine?: { value: string };
        cwd?: { fsPath: string };
        exitCode?: number;
        read?: () => vscode.ReadableStream<string>;
    };
}

interface ShellExecutionEndEvent {
    terminal: vscode.Terminal;
    shellIntegration?: {
        executeCommand: {
            exitCode?: number;
        };
    };
    execution?: {
        exitCode?: number;
    };
    exitCode?: number;
}

interface PendingExecution {
    command: string;
    terminalName: string;
    workingDir?: string;
    startTime: number;
    outputLines: string[];
}

/**
 * Singleton service for terminal history tracking
 */
export class TerminalHistoryService implements vscode.Disposable {
    private static _instance: TerminalHistoryService | null = null;

    private _history: TerminalHistoryEntry[] = [];
    private _maxHistorySize = 100;
    private _maxOutputLines = 50;
    private _disposables: vscode.Disposable[] = [];
    private _onHistoryChanged = new vscode.EventEmitter<void>();
    private _pendingExecutions = new Map<string, PendingExecution>();
    private _shellIntegrationAvailable = false;

    readonly onHistoryChanged = this._onHistoryChanged.event;

    private constructor() {
        // Check for shell integration API availability (VS Code 1.93+)
        this._initShellIntegration();

        // Track terminal lifecycle
        this._disposables.push(
            vscode.window.onDidOpenTerminal(terminal => {
                this._trackTerminal(terminal);
            }),
            vscode.window.onDidCloseTerminal(_terminal => {
                // Terminal closed - we keep the history
            })
        );

        // Track existing terminals
        vscode.window.terminals.forEach(terminal => {
            this._trackTerminal(terminal);
        });
    }

    static getInstance(): TerminalHistoryService {
        if (!TerminalHistoryService._instance) {
            TerminalHistoryService._instance = new TerminalHistoryService();
        }
        return TerminalHistoryService._instance;
    }

    dispose(): void {
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
        this._pendingExecutions.clear();
        this._onHistoryChanged.dispose();
        TerminalHistoryService._instance = null;
    }

    /**
     * Check if shell integration is available
     */
    get shellIntegrationAvailable(): boolean {
        return this._shellIntegrationAvailable;
    }

    /**
     * Add a command to history (called by terminal agent or other sources)
     */
    addCommand(entry: Omit<TerminalHistoryEntry, 'timestamp'>): void {
        const fullEntry: TerminalHistoryEntry = {
            ...entry,
            timestamp: Date.now(),
        };

        this._history.unshift(fullEntry);

        // Trim to max size
        if (this._history.length > this._maxHistorySize) {
            this._history = this._history.slice(0, this._maxHistorySize);
        }

        this._onHistoryChanged.fire();
    }

    /**
     * Get recent commands for context
     */
    getRecentCommands(limit = 10): TerminalHistoryEntry[] {
        return this._history.slice(0, limit);
    }

    /**
     * Get commands for a specific terminal
     */
    getTerminalCommands(terminalName: string, limit = 10): TerminalHistoryEntry[] {
        return this._history
            .filter(entry => entry.terminalName === terminalName)
            .slice(0, limit);
    }

    /**
     * Get formatted context string for AI prompts
     */
    getContextString(limit = 5): string {
        const commands = this.getRecentCommands(limit);

        if (commands.length === 0) {
            return '// No recent terminal commands';
        }

        const lines = ['// Recent terminal commands:'];

        for (const entry of commands) {
            const timeAgo = this._formatTimeAgo(entry.timestamp);
            lines.push(`// [${timeAgo}] $ ${entry.command}`);

            if (entry.exitCode !== undefined && entry.exitCode !== 0) {
                lines.push(`//   Exit code: ${entry.exitCode}`);
            }

            if (entry.output) {
                // Include truncated output
                const outputLines = entry.output.split('\n').slice(0, 3);
                for (const line of outputLines) {
                    if (line.trim()) {
                        lines.push(`//   > ${line.slice(0, 80)}`);
                    }
                }
                if (entry.output.split('\n').length > 3) {
                    lines.push('//   ...');
                }
            }
        }

        return lines.join('\n');
    }

    /**
     * Clear all history
     */
    clear(): void {
        this._history = [];
        this._onHistoryChanged.fire();
    }

    /**
     * Get total history count
     */
    get count(): number {
        return this._history.length;
    }

    // --- Private Methods ---

    /**
     * Initialize VS Code Shell Integration API if available
     */
    private _initShellIntegration(): void {
        try {
            // Check if shell integration events exist (VS Code 1.93+)
            // These are proposed APIs that may not be available in all versions
            if ('onDidStartTerminalShellExecution' in vscode.window &&
                'onDidEndTerminalShellExecution' in vscode.window) {

                this._shellIntegrationAvailable = true;

                // Listen for command execution start
                const startEvent = (vscode.window as any).onDidStartTerminalShellExecution;
                if (startEvent) {
                    this._disposables.push(
                        startEvent((event: ShellExecutionStartEvent) => {
                            this._onShellExecutionStart(event);
                        })
                    );
                }

                // Listen for command execution end
                const endEvent = (vscode.window as any).onDidEndTerminalShellExecution;
                if (endEvent) {
                    this._disposables.push(
                        endEvent((event: ShellExecutionEndEvent) => {
                            this._onShellExecutionEnd(event);
                        })
                    );
                }

                console.log('[TerminalHistory] Shell integration API available');
            } else {
                console.log('[TerminalHistory] Shell integration API not available, using manual tracking');
            }
        } catch (error) {
            console.log('[TerminalHistory] Error initializing shell integration:', error);
        }
    }

    /**
     * Handle shell execution start event
     */
    private _onShellExecutionStart(event: ShellExecutionStartEvent): void {
        try {
            const terminal = event.terminal as vscode.Terminal;
            const execution = event.shellIntegration?.executeCommand || event.execution;

            if (!terminal || !execution) {
                return;
            }

            const commandLine = execution.commandLine?.value || execution.commandLine || '';
            const cwd = execution.cwd?.fsPath || undefined;

            if (!commandLine) {
                return;
            }

            // Track this pending execution
            const key = `${terminal.name}-${Date.now()}`;
            this._pendingExecutions.set(key, {
                command: commandLine,
                terminalName: terminal.name,
                workingDir: cwd,
                startTime: Date.now(),
                outputLines: [],
            });

            // Also try to capture output stream if available
            if (execution.read) {
                this._captureOutput(key, execution);
            }
        } catch (error) {
            console.error('[TerminalHistory] Error handling execution start:', error);
        }
    }

    /**
     * Handle shell execution end event
     */
    private _onShellExecutionEnd(event: ShellExecutionEndEvent): void {
        try {
            const terminal = event.terminal as vscode.Terminal;
            const execution = event.shellIntegration?.executeCommand || event.execution;
            const exitCode = event.exitCode ?? execution?.exitCode;

            if (!terminal) {
                return;
            }

            const commandLine = execution?.commandLine?.value || execution?.commandLine || '';

            // Find matching pending execution
            let matchingKey: string | undefined;
            for (const [key, pending] of this._pendingExecutions) {
                if (pending.terminalName === terminal.name &&
                    (pending.command === commandLine || commandLine === '')) {
                    matchingKey = key;
                    break;
                }
            }

            if (matchingKey) {
                const pending = this._pendingExecutions.get(matchingKey)!;
                this._pendingExecutions.delete(matchingKey);

                // Add to history with captured output
                this.addCommand({
                    command: pending.command,
                    terminalName: pending.terminalName,
                    workingDir: pending.workingDir,
                    exitCode: typeof exitCode === 'number' ? exitCode : undefined,
                    output: pending.outputLines.length > 0
                        ? this._processOutput(pending.outputLines)
                        : undefined,
                });
            } else if (commandLine) {
                // No pending execution found, create new entry
                this.addCommand({
                    command: commandLine,
                    terminalName: terminal.name,
                    exitCode: typeof exitCode === 'number' ? exitCode : undefined,
                });
            }
        } catch (error) {
            console.error('[TerminalHistory] Error handling execution end:', error);
        }
    }

    /**
     * Capture output from shell execution stream
     */
    private async _captureOutput(key: string, execution: ShellExecutionStartEvent['shellIntegration']['executeCommand']): Promise<void> {
        try {
            const pending = this._pendingExecutions.get(key);
            if (!pending) {
                return;
            }

            // Read output stream if available
            const reader = execution.read();
            if (!reader) {
                return;
            }

            for await (const data of reader) {
                if (!this._pendingExecutions.has(key)) {
                    break; // Execution already completed
                }

                const text = typeof data === 'string' ? data : data.toString();
                const lines = text.split('\n');

                for (const line of lines) {
                    if (pending.outputLines.length < this._maxOutputLines) {
                        pending.outputLines.push(line);
                    }
                }
            }
        } catch (error) {
            // Stream reading may fail, ignore errors
        }
    }

    /**
     * Process captured output to clean up terminal artifacts
     */
    private _processOutput(lines: string[]): string {
        // Remove ANSI escape codes and shell integration markers
        const cleanLines = lines
            .map(line => this._cleanLine(line))
            .filter(line => line.trim().length > 0);

        // Collapse repeated lines (e.g., progress bars)
        const deduped: string[] = [];
        let lastLine = '';

        for (const line of cleanLines) {
            if (line !== lastLine) {
                deduped.push(line);
                lastLine = line;
            }
        }

        // Truncate if too long (keep 20% beginning, 80% end like Cline)
        if (deduped.length > this._maxOutputLines) {
            const keepStart = Math.floor(this._maxOutputLines * 0.2);
            const keepEnd = this._maxOutputLines - keepStart - 1;
            return [
                ...deduped.slice(0, keepStart),
                '... (truncated) ...',
                ...deduped.slice(-keepEnd),
            ].join('\n');
        }

        return deduped.join('\n');
    }

    /**
     * Clean a single line of terminal output
     */
    private _cleanLine(line: string): string {
        // Remove ANSI escape codes
        let cleaned = line.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');

        // Remove shell integration escape sequences (OSC 633/133)
        cleaned = cleaned.replace(/\x1b\][0-9]+;[^\x07]*\x07/g, '');

        // Process carriage returns (keep only final state)
        if (cleaned.includes('\r')) {
            const parts = cleaned.split('\r');
            cleaned = parts[parts.length - 1];
        }

        // Remove backspaces
        while (cleaned.includes('\b')) {
            cleaned = cleaned.replace(/[^\b]\b/g, '');
        }

        return cleaned;
    }

    private _trackTerminal(_terminal: vscode.Terminal): void {
        // Terminal tracking is now handled via shell integration events
        // For terminals without shell integration, commands are added via addCommand()
    }

    private _formatTimeAgo(timestamp: number): string {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);

        if (seconds < 60) {
            return 'just now';
        }
        if (seconds < 3600) {
            const mins = Math.floor(seconds / 60);
            return `${mins}m ago`;
        }
        if (seconds < 86400) {
            const hours = Math.floor(seconds / 3600);
            return `${hours}h ago`;
        }
        const days = Math.floor(seconds / 86400);
        return `${days}d ago`;
    }
}
