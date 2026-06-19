// Unit tests for the StateStore (the extension's central state singleton). This also
// exercises the `vscode` mock (the store uses vscode.EventEmitter), proving the unit
// harness can cover vscode-dependent modules without the electron test host.
import { describe, it, expect, beforeEach } from 'vitest';
import { getStore, StateStore, selectors } from '../state/store';

beforeEach(() => {
    StateStore.resetInstance();
});

describe('StateStore', () => {
    it('getStore() returns a stable singleton', () => {
        expect(getStore()).toBe(getStore());
        expect(getStore()).toBe(StateStore.getInstance());
    });

    it('resetInstance() yields a fresh instance', () => {
        const first = getStore();
        StateStore.resetInstance();
        expect(getStore()).not.toBe(first);
    });

    it('starts with a sane, uninitialized initial state', () => {
        const state = getStore().getState();
        expect(state.initialized).toBe(false);
        expect(state.server).toBeDefined();
        expect(state.session).toBeDefined();
        expect(state.settings).toBeDefined();
    });

    it('updateState() deep-merges without clobbering sibling slices', () => {
        const store = getStore();
        const serverBefore = store.getState().server.status;
        store.updateState({ session: { mode: 'plan' } } as never);
        expect(store.getState().session.mode).toBe('plan');
        // The server slice must be untouched by a session-only update.
        expect(store.getState().server.status).toBe(serverBefore);
    });

    it('emits onStateChange when state is updated', () => {
        const store = getStore();
        let fired = 0;
        const sub = store.onStateChange(() => { fired++; });
        store.updateState({ session: { mode: 'review' } } as never);
        expect(fired).toBeGreaterThan(0);
        sub.dispose();
    });

    it('selectors read the expected slices', () => {
        const store = getStore();
        store.updateState({ session: { mode: 'explore' } } as never);
        expect(selectors.mode(store.getState())).toBe('explore');
        expect(selectors.isInitialized(store.getState())).toBe(false);
    });
});
