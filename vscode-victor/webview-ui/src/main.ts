/**
 * Victor Chat WebView - Main Entry Point
 */

import './styles/theme.css';
import App from './components/App.svelte';

const app = new App({
  target: document.getElementById('app')!,
});

export default app;
