/**
 * Svelte store for settings management
 */

import { writable, type Writable } from 'svelte/store';
import type { Settings } from './types';

const defaultSettings: Settings = {
  provider: 'anthropic',
  model: 'claude-sonnet-4-20250514',
  mode: 'build',
  theme: 'dark',
};

function createSettingsStore() {
  const { subscribe, set, update }: Writable<Settings> = writable(defaultSettings);

  return {
    subscribe,

    init(settings: Partial<Settings>) {
      update(state => ({
        ...state,
        ...settings,
      }));
    },

    setProvider(provider: string) {
      update(state => ({ ...state, provider }));
    },

    setModel(model: string) {
      update(state => ({ ...state, model }));
    },

    setMode(mode: Settings['mode']) {
      update(state => ({ ...state, mode }));
    },

    setTheme(theme: Settings['theme']) {
      update(state => ({ ...state, theme }));
    },

    reset() {
      set(defaultSettings);
    },
  };
}

export const settingsStore = createSettingsStore();
