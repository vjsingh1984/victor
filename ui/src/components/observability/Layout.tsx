/**
 * Observability Dashboard Layout
 *
 * Provides navigation and layout for observability pages:
 * - Live Dashboard
 * - Event Browser
 * - Session Viewer (Phase 3)
 * - Metrics View (Phase 2)
 * - Trace Explorer (Phase 4)
 */

import { Link, useLocation } from 'react-router-dom';
import './Layout.css';
import { Routes, Route } from 'react-router-dom';
import { LiveDashboard } from './LiveDashboard';
import { EventBrowser } from './EventBrowser';

interface LayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <div className="observability-dashboard">
      <Sidebar currentPath={location.pathname} />
      <div className="main-content">
        <TopBar />
        <div className="content-area">
          {children}
        </div>
      </div>
    </div>
  );
}

interface SidebarProps {
  currentPath: string;
}

function Sidebar({ currentPath }: SidebarProps) {
  const navigation = [
    { path: '/obs/live', label: 'Live Dashboard', icon: '📊' },
    { path: '/obs/events', label: 'Event Browser', icon: '🔍' },
    { path: '/obs/sessions', label: 'Sessions', icon: '💬', disabled: true },
    { path: '/obs/metrics', label: 'Metrics', icon: '📈', disabled: true },
    { path: '/obs/traces', label: 'Traces', icon: '🌳', disabled: true },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Victor Observability</h2>
      </div>
      <nav className="sidebar-nav">
        {navigation.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-item ${
              currentPath === item.path ? 'active' : ''
            } ${item.disabled ? 'disabled' : ''}`}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
            {item.disabled && (
              <span className="nav-badge">Soon</span>
            )}
          </Link>
        ))}
      </nav>
      <div className="sidebar-footer">
        <a
          href="https://github.com/victor-ai/victor"
          target="_blank"
          rel="noopener noreferrer"
          className="footer-link"
        >
          Documentation
        </a>
      </div>
    </aside>
  );
}

function TopBar() {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1>Observability</h1>
      </div>
      <div className="topbar-right">
        <div className="status-indicators">
          <span className="status-item" title="API Status">
            <span className="status-dot status-ok" />
            <span className="status-text">API OK</span>
          </span>
          <span className="status-item" title="WebSocket Status">
            <span className="status-dot status-ok" />
            <span className="status-text">WS OK</span>
          </span>
        </div>
      </div>
    </header>
  );
}

/**
 * Main observability routes component
 *
 * Usage in App.tsx:
 *   <Routes>
 *     <Route path="/obs/*" element={<ObservabilityRoutes />} />
 *   </Routes>
 */
export function ObservabilityRoutes() {
  return (
    <DashboardLayout>
      <Routes>
        <Route path="/live" element={<LiveDashboard />} />
        <Route path="/events" element={<EventBrowser />} />
        <Route
          path="/sessions"
          element={<div className="coming-soon">Session Viewer - Coming in Phase 3</div>}
        />
        <Route
          path="/metrics"
          element={<div className="coming-soon">Metrics View - Coming in Phase 2</div>}
        />
        <Route
          path="/traces"
          element={<div className="coming-soon">Trace Explorer - Coming in Phase 4</div>}
        />
        <Route path="/" element={<LiveDashboard />} />
      </Routes>
    </DashboardLayout>
  );
}
