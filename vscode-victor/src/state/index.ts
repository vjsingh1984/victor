/**
 * State Module
 *
 * Centralized state management for the Victor VS Code extension.
 * Replaces distributed module-level state variables with a single source of truth.
 *
 * Usage:
 * ```typescript
 * import { getStore, selectors } from './state';
 *
 * // Get the store instance
 * const store = getStore();
 *
 * // Read state
 * const mode = store.select(selectors.mode);
 *
 * // Update state
 * store.setMode('plan');
 *
 * // Subscribe to changes
 * const disposable = store.subscribe('session', (state) => {
 *     console.log('Session changed:', state.session);
 * });
 *
 * // Subscribe with selector
 * const disposable2 = store.subscribeToSelector(selectors.mode, (mode) => {
 *     updateStatusBar(mode);
 * });
 * ```
 */

export { StateStore, getStore, selectors } from './store';
export * from './types';
