/**
 * Main entry point with routing for Victor UI
 *
 * Provides routing between:
 * - Chat interface (root path)
 * - Observability dashboard (/obs/*)
 */

import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import App from './App';
import { ObservabilityRoutes } from './components/observability';

function RouterApp() {
  return (
    <BrowserRouter>
      <div className="victor-app">
        {/* Navigation */}
        <nav className="app-nav">
          <Link to="/" className="nav-link">
            💬 Chat
          </Link>
          <Link to="/obs/live" className="nav-link">
            📊 Observability
          </Link>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/obs/*" element={<ObservabilityRoutes />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default RouterApp;
