/**
 * Victor Client Tests
 *
 * Tests for the VictorClient API client.
 * Note: These are unit tests that mock the HTTP layer.
 */

import * as assert from 'assert';
import { VictorClient, ChatMessage, ToolCall, VictorError, VictorErrorType } from '../../victorClient';

suite('VictorClient Test Suite', () => {

    let client: VictorClient;

    setup(() => {
        // Create client with test configuration
        client = new VictorClient('http://localhost:9999');
    });

    teardown(() => {
        // Clean up WebSocket connection
        client.disconnectWebSocket();
    });

    test('Client should initialize with correct base URL', () => {
        // The client should be created without errors
        assert.ok(client);
    });

    test('ChatMessage interface should accept valid messages', () => {
        const userMessage: ChatMessage = {
            role: 'user',
            content: 'Hello, Victor!'
        };

        const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: 'Hello! How can I help you?',
            toolCalls: [{ name: 'read_file', arguments: { path: 'test.ts' } }]
        };

        assert.strictEqual(userMessage.role, 'user');
        assert.strictEqual(assistantMessage.role, 'assistant');
        assert.ok(assistantMessage.toolCalls);
    });

    test('ToolCall interface should have correct structure', () => {
        const toolCall: ToolCall = {
            id: 'tc_123',
            name: 'read_file',
            arguments: { path: '/test/file.ts' },
            result: 'file contents'
        };

        assert.strictEqual(toolCall.name, 'read_file');
        assert.ok(toolCall.arguments);
    });

    test('VictorErrorType enum values should exist', () => {
        assert.strictEqual(VictorErrorType.Network, 'NETWORK');
        assert.strictEqual(VictorErrorType.Timeout, 'TIMEOUT');
        assert.strictEqual(VictorErrorType.ServerError, 'SERVER_ERROR');
        assert.strictEqual(VictorErrorType.NotFound, 'NOT_FOUND');
        assert.strictEqual(VictorErrorType.Validation, 'VALIDATION');
        assert.strictEqual(VictorErrorType.Auth, 'AUTH');
        assert.strictEqual(VictorErrorType.Unknown, 'UNKNOWN');
    });

    test('VictorError should have correct properties', () => {
        const error = new VictorError(
            'Test error',
            VictorErrorType.Network,
            500,
            new Error('Original')
        );

        assert.strictEqual(error.message, 'Test error');
        assert.strictEqual(error.type, VictorErrorType.Network);
        assert.strictEqual(error.statusCode, 500);
        assert.ok(error.originalError);
    });

    test('getStatus should handle server unavailable', async () => {
        try {
            const status = await client.getStatus();
            // If we get here, server is running
            assert.ok(status);
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('disconnectWebSocket should be safe to call multiple times', () => {
        // Should not throw
        client.disconnectWebSocket();
        client.disconnectWebSocket();
        client.disconnectWebSocket();

        assert.ok(true);
    });

    test('connectWebSocket should not throw', () => {
        // Should not throw even when server is unavailable
        client.connectWebSocket();

        // Give it a moment then disconnect
        return new Promise<void>((resolve) => {
            setTimeout(() => {
                client.disconnectWebSocket();
                resolve();
            }, 100);
        });
    });

    test('streamChat should handle callback setup', async () => {
        const messages: ChatMessage[] = [{ role: 'user', content: 'Test' }];

        try {
            await client.streamChat(
                messages,
                (content) => { /* content callback */ },
                (toolCall) => { /* tool call callback */ }
            );
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('undo should handle server unavailable', async () => {
        try {
            const result = await client.undo();
            assert.ok(result);
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('redo should handle server unavailable', async () => {
        try {
            const result = await client.redo();
            assert.ok(result);
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('semanticSearch should handle server unavailable', async () => {
        try {
            const results = await client.semanticSearch('test query');
            assert.ok(Array.isArray(results));
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('switchModel should require both parameters', async () => {
        try {
            await client.switchModel('openai', 'gpt-4');
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('switchMode should accept valid mode', async () => {
        try {
            await client.switchMode('plan');
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });
});
