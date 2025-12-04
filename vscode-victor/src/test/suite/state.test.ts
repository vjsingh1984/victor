/**
 * State Management Tests
 *
 * Tests for the centralized state store and state management.
 */

import * as assert from 'assert';
import { StateStore, getStore, selectors } from '../../state/store';
import { ConnectionState, AgentMode } from '../../state/types';
import { ServerStatus } from '../../serverManager';

suite('State Management Test Suite', () => {

    setup(() => {
        // Reset the singleton instance for each test
        StateStore.resetInstance();
    });

    test('StateStore singleton should return same instance', () => {
        const store1 = StateStore.getInstance();
        const store2 = StateStore.getInstance();

        assert.strictEqual(store1, store2);
    });

    test('getStore should return StateStore instance', () => {
        const store = getStore();
        assert.ok(store instanceof StateStore);
    });

    test('Initial state should have correct structure', () => {
        const store = getStore();
        const state = store.getState();

        // Check server state exists
        assert.ok(state.server);
        assert.strictEqual(state.server.status, ServerStatus.Stopped);
        assert.strictEqual(state.server.connectionState, ConnectionState.Disconnected);

        // Check session state exists
        assert.ok(state.session);
        assert.strictEqual(state.session.mode, 'build');
        assert.ok(state.session.model);
        assert.ok(state.session.conversation);

        // Check UI state exists
        assert.ok(state.ui);

        // Check settings state exists
        assert.ok(state.settings);
    });

    test('updateState should update nested state', () => {
        const store = getStore();

        store.updateState({
            server: { status: ServerStatus.Running }
        });

        const state = store.getState();
        assert.strictEqual(state.server.status, ServerStatus.Running);
    });

    test('updateState should preserve unmodified fields', () => {
        const store = getStore();
        const originalPort = store.getState().server.port;

        store.updateState({
            server: { status: ServerStatus.Running }
        });

        const state = store.getState();
        assert.strictEqual(state.server.port, originalPort);
    });

    test('setServerStatus convenience method should work', () => {
        const store = getStore();

        store.setServerStatus(ServerStatus.Starting);
        assert.strictEqual(store.getState().server.status, ServerStatus.Starting);

        store.setServerStatus(ServerStatus.Running);
        assert.strictEqual(store.getState().server.status, ServerStatus.Running);
    });

    test('setConnectionState should update connection and error', () => {
        const store = getStore();

        store.setConnectionState(ConnectionState.Error, 'Test error');

        const state = store.getState();
        assert.strictEqual(state.server.connectionState, ConnectionState.Error);
        assert.strictEqual(state.server.lastError, 'Test error');
    });

    test('subscribe should notify on state changes', () => {
        const store = getStore();
        let notified = false;

        const disposable = store.subscribe('server', () => {
            notified = true;
        });

        store.setServerStatus(ServerStatus.Running);

        assert.strictEqual(notified, true);
        disposable.dispose();
    });

    test('unsubscribe should stop notifications', () => {
        const store = getStore();
        let notifyCount = 0;

        const disposable = store.subscribe('server', () => {
            notifyCount++;
        });

        store.setServerStatus(ServerStatus.Running);
        assert.strictEqual(notifyCount, 1);

        disposable.dispose();

        store.setServerStatus(ServerStatus.Stopped);
        assert.strictEqual(notifyCount, 1); // Should not increase
    });

    test('select should return correct value', () => {
        const store = getStore();

        const status = store.select(selectors.serverStatus);
        assert.strictEqual(status, ServerStatus.Stopped);

        const mode = store.select(selectors.mode);
        assert.strictEqual(mode, 'build');
    });

    test('startNewConversation should reset conversation', () => {
        const store = getStore();
        const originalId = store.getState().session.conversation.id;

        store.startNewConversation();

        const newId = store.getState().session.conversation.id;
        assert.notStrictEqual(newId, originalId);
        assert.strictEqual(store.getState().session.conversation.messages.length, 0);
    });

    test('setStreaming should update streaming state', () => {
        const store = getStore();

        store.setStreaming(true);
        assert.strictEqual(store.getState().session.conversation.isStreaming, true);

        store.setStreaming(false);
        assert.strictEqual(store.getState().session.conversation.isStreaming, false);
    });

    test('ConnectionState enum values should be correct', () => {
        assert.strictEqual(ConnectionState.Disconnected, 'disconnected');
        assert.strictEqual(ConnectionState.Connecting, 'connecting');
        assert.strictEqual(ConnectionState.Connected, 'connected');
        assert.strictEqual(ConnectionState.Reconnecting, 'reconnecting');
        assert.strictEqual(ConnectionState.Error, 'error');
    });

    test('ServerStatus enum values should be correct', () => {
        assert.strictEqual(ServerStatus.Stopped, 'stopped');
        assert.strictEqual(ServerStatus.Starting, 'starting');
        assert.strictEqual(ServerStatus.Running, 'running');
        assert.strictEqual(ServerStatus.Error, 'error');
    });

    test('dispose should clean up resources', () => {
        const store = getStore();

        // Should not throw
        store.dispose();
        assert.ok(true);
    });
});
