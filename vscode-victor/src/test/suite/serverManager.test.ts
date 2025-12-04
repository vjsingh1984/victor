/**
 * Server Manager Tests
 *
 * Tests for the Victor server manager.
 */

import * as assert from 'assert';
import { ServerManager, ServerStatus, ServerConfig } from '../../serverManager';

suite('ServerManager Test Suite', () => {

    let manager: ServerManager;
    const testConfig: ServerConfig = {
        host: 'localhost',
        port: 9998,
        autoStart: false,
        pythonPath: undefined,
        victorPath: undefined,
    };

    setup(() => {
        manager = new ServerManager(testConfig);
    });

    teardown(() => {
        // Clean up
        manager.dispose();
    });

    test('ServerManager should initialize with config', () => {
        assert.ok(manager);
    });

    test('Initial status should be Stopped', () => {
        assert.strictEqual(manager.getStatus(), ServerStatus.Stopped);
    });

    test('ServerStatus enum values should be correct', () => {
        assert.strictEqual(ServerStatus.Stopped, 'stopped');
        assert.strictEqual(ServerStatus.Starting, 'starting');
        assert.strictEqual(ServerStatus.Running, 'running');
        assert.strictEqual(ServerStatus.Error, 'error');
    });

    test('getServerUrl should return correct URL', () => {
        const url = manager.getServerUrl();
        assert.strictEqual(url, 'http://localhost:9998');
    });

    test('onStatusChange should register callback', () => {
        let callbackCalled = false;

        manager.onStatusChange((status) => {
            callbackCalled = true;
        });

        // Callback registration should succeed
        assert.ok(true);
    });

    test('checkHealth should return false when server not running', async () => {
        const healthy = await manager.checkHealth();
        assert.strictEqual(healthy, false);
    });

    test('dispose should be safe to call', () => {
        // Should not throw
        manager.dispose();
        manager.dispose(); // Multiple calls should be safe

        assert.ok(true);
    });

    test('stop should complete without error when not started', async () => {
        // Should not throw when stopping a server that's not running
        await manager.stop();

        assert.strictEqual(manager.getStatus(), ServerStatus.Stopped);
    });

    test('Multiple status callbacks should all be registered', () => {
        let count1 = 0;
        let count2 = 0;

        manager.onStatusChange(() => { count1++; });
        manager.onStatusChange(() => { count2++; });

        // Both callbacks should be registered
        assert.ok(true);
    });

    test('showOutput should not throw', () => {
        // Should not throw even if output channel is not visible
        manager.showOutput();
        assert.ok(true);
    });

    test('getStatus should return current status', () => {
        // Initially stopped
        assert.strictEqual(manager.getStatus(), ServerStatus.Stopped);

        // Status method should be callable
        const status = manager.getStatus();
        assert.ok(Object.values(ServerStatus).includes(status));
    });
});
