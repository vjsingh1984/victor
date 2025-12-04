/**
 * Victor Server Manager
 *
 * Handles automatic spawning, lifecycle management, and health monitoring
 * of the Victor Python backend server.
 */

import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

export interface ServerConfig {
    host: string;
    port: number;
    autoStart: boolean;
    pythonPath?: string;
    victorPath?: string;
}

export enum ServerStatus {
    Stopped = 'stopped',
    Starting = 'starting',
    Running = 'running',
    Error = 'error'
}

export class ServerManager {
    private serverProcess: cp.ChildProcess | null = null;
    private status: ServerStatus = ServerStatus.Stopped;
    private statusChangeCallbacks: ((status: ServerStatus) => void)[] = [];
    private config: ServerConfig;
    private outputChannel: vscode.OutputChannel;
    private healthCheckInterval: NodeJS.Timeout | null = null;

    constructor(config: ServerConfig) {
        this.config = config;
        this.outputChannel = vscode.window.createOutputChannel('Victor Server');
    }

    /**
     * Get the current server status
     */
    getStatus(): ServerStatus {
        return this.status;
    }

    /**
     * Register a callback for status changes
     */
    onStatusChange(callback: (status: ServerStatus) => void): void {
        this.statusChangeCallbacks.push(callback);
    }

    /**
     * Get the server URL
     */
    getServerUrl(): string {
        return `http://${this.config.host}:${this.config.port}`;
    }

    /**
     * Check if server is running by hitting health endpoint
     */
    async checkHealth(): Promise<boolean> {
        try {
            const response = await fetch(`${this.getServerUrl()}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(2000)
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    /**
     * Start the server
     */
    async start(): Promise<boolean> {
        // First check if server is already running
        if (await this.checkHealth()) {
            this.setStatus(ServerStatus.Running);
            this.log('Server already running');
            return true;
        }

        this.setStatus(ServerStatus.Starting);
        this.log('Starting Victor server...');

        // Find the victor executable
        const victorCommand = await this.findVictorCommand();
        if (!victorCommand) {
            this.setStatus(ServerStatus.Error);
            this.log('ERROR: Could not find victor command');
            vscode.window.showErrorMessage(
                'Victor not found. Please install with: pip install -e ".[dev]"'
            );
            return false;
        }

        this.log(`Using command: ${victorCommand}`);

        // Build spawn arguments
        const args = [
            'serve',
            '--host', this.config.host,
            '--port', this.config.port.toString(),
            '--log-level', 'INFO'
        ];

        try {
            // Spawn the server process
            this.serverProcess = cp.spawn(victorCommand, args, {
                shell: true,
                env: {
                    ...process.env,
                    PYTHONUNBUFFERED: '1'
                },
                cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
            });

            // Handle stdout
            this.serverProcess.stdout?.on('data', (data: Buffer) => {
                this.log(data.toString().trim());
            });

            // Handle stderr
            this.serverProcess.stderr?.on('data', (data: Buffer) => {
                this.log(`[stderr] ${data.toString().trim()}`);
            });

            // Handle process exit
            this.serverProcess.on('exit', (code, signal) => {
                this.log(`Server process exited (code: ${code}, signal: ${signal})`);
                this.serverProcess = null;
                this.setStatus(ServerStatus.Stopped);
                this.stopHealthCheck();
            });

            // Handle errors
            this.serverProcess.on('error', (err) => {
                this.log(`ERROR: ${err.message}`);
                this.setStatus(ServerStatus.Error);
            });

            // Wait for server to be ready
            const ready = await this.waitForServer(15000);
            if (ready) {
                this.setStatus(ServerStatus.Running);
                this.log('Server started successfully');
                this.startHealthCheck();
                return true;
            } else {
                this.setStatus(ServerStatus.Error);
                this.log('ERROR: Server failed to start within timeout');
                await this.stop();
                return false;
            }

        } catch (error) {
            this.setStatus(ServerStatus.Error);
            this.log(`ERROR: Failed to spawn server: ${error}`);
            return false;
        }
    }

    /**
     * Stop the server
     */
    async stop(): Promise<void> {
        this.stopHealthCheck();

        if (this.serverProcess) {
            this.log('Stopping Victor server...');

            // Try graceful shutdown via API first
            try {
                await fetch(`${this.getServerUrl()}/shutdown`, {
                    method: 'POST',
                    signal: AbortSignal.timeout(2000)
                });
            } catch {
                // Ignore errors, will force kill below
            }

            // Wait a bit for graceful shutdown
            await this.sleep(500);

            // Force kill if still running
            if (this.serverProcess && !this.serverProcess.killed) {
                this.serverProcess.kill('SIGTERM');
                await this.sleep(500);

                if (this.serverProcess && !this.serverProcess.killed) {
                    this.serverProcess.kill('SIGKILL');
                }
            }

            this.serverProcess = null;
        }

        this.setStatus(ServerStatus.Stopped);
        this.log('Server stopped');
    }

    /**
     * Restart the server
     */
    async restart(): Promise<boolean> {
        await this.stop();
        return this.start();
    }

    /**
     * Dispose of resources
     *
     * Note: This performs a synchronous cleanup. The async stop() is called
     * but not awaited to comply with VS Code's Disposable interface.
     * The server process is killed immediately if running.
     */
    dispose(): void {
        // Stop health check first
        this.stopHealthCheck();

        // Kill the server process immediately (sync cleanup)
        if (this.serverProcess && !this.serverProcess.killed) {
            try {
                this.serverProcess.kill('SIGTERM');
            } catch {
                // Ignore errors during cleanup
            }
            this.serverProcess = null;
        }

        this.setStatus(ServerStatus.Stopped);
        this.outputChannel.dispose();
    }

    /**
     * Show the output channel
     */
    showOutput(): void {
        this.outputChannel.show();
    }

    // --- Private methods ---

    private setStatus(status: ServerStatus): void {
        if (this.status !== status) {
            this.status = status;
            for (const callback of this.statusChangeCallbacks) {
                callback(status);
            }
        }
    }

    private log(message: string): void {
        const timestamp = new Date().toISOString();
        this.outputChannel.appendLine(`[${timestamp}] ${message}`);
    }

    /**
     * Find the victor command
     * Tries: victor in PATH, python -m victor.ui.cli, bundled binary
     */
    private async findVictorCommand(): Promise<string | null> {
        // 1. Check if victor is in PATH
        if (await this.commandExists('victor')) {
            return 'victor';
        }

        // 2. Check if vic (alias) is in PATH
        if (await this.commandExists('vic')) {
            return 'vic';
        }

        // 3. Check configured Python path
        if (this.config.pythonPath) {
            return `${this.config.pythonPath} -m victor.ui.cli`;
        }

        // 4. Try python3 -m victor.ui.cli
        if (await this.commandExists('python3')) {
            // Verify victor module is installed
            try {
                const result = cp.execSync('python3 -c "import victor"', { timeout: 5000 });
                return 'python3 -m victor.ui.cli';
            } catch {
                // Victor not installed
            }
        }

        // 5. Try python -m victor.ui.cli
        if (await this.commandExists('python')) {
            try {
                const result = cp.execSync('python -c "import victor"', { timeout: 5000 });
                return 'python -m victor.ui.cli';
            } catch {
                // Victor not installed
            }
        }

        // 6. Check for bundled binary in extension
        if (this.config.victorPath && fs.existsSync(this.config.victorPath)) {
            return this.config.victorPath;
        }

        return null;
    }

    /**
     * Check if a command exists in PATH
     */
    private async commandExists(command: string): Promise<boolean> {
        try {
            const checkCmd = process.platform === 'win32' ? 'where' : 'which';
            cp.execSync(`${checkCmd} ${command}`, { stdio: 'ignore', timeout: 2000 });
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Wait for server to respond to health checks
     */
    private async waitForServer(timeoutMs: number): Promise<boolean> {
        const startTime = Date.now();
        const checkInterval = 500;

        while (Date.now() - startTime < timeoutMs) {
            if (await this.checkHealth()) {
                return true;
            }
            await this.sleep(checkInterval);
        }

        return false;
    }

    /**
     * Start periodic health checks
     */
    private startHealthCheck(): void {
        this.stopHealthCheck();
        this.healthCheckInterval = setInterval(async () => {
            const healthy = await this.checkHealth();
            if (!healthy && this.status === ServerStatus.Running) {
                this.log('WARNING: Health check failed');
                this.setStatus(ServerStatus.Error);
            } else if (healthy && this.status === ServerStatus.Error) {
                this.log('Server recovered');
                this.setStatus(ServerStatus.Running);
            }
        }, 30000); // Check every 30 seconds
    }

    /**
     * Stop periodic health checks
     */
    private stopHealthCheck(): void {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
    }

    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Create a server manager with VS Code configuration
 */
export function createServerManager(): ServerManager {
    const config = vscode.workspace.getConfiguration('victor');

    return new ServerManager({
        host: '127.0.0.1',
        port: config.get('serverPort', 8765),
        autoStart: config.get('autoStart', true),
        pythonPath: config.get('pythonPath'),
        victorPath: config.get('victorPath')
    });
}
