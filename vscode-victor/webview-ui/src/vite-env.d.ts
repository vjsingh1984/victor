/// <reference types="svelte" />
/// <reference types="vite/client" />

// Declare acquireVsCodeApi as a global function
declare function acquireVsCodeApi(): {
  postMessage(message: unknown): void;
  getState(): unknown;
  setState(state: unknown): void;
};
