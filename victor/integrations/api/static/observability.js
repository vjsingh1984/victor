// Victor Observability Dashboard JavaScript

const API_BASE = '/observability';
const POLL_INTERVAL = 5000; // 5 seconds

let currentTab = 'dashboard';
let pollTimer = null;

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        switchTab(tabName);
    });
});

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });

    currentTab = tabName;
    refreshData();
}

// Data fetching
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        return null;
    }
}

// Refresh all data based on current tab
async function refreshData() {
    updateConnectionStatus(true);

    switch (currentTab) {
        case 'dashboard':
            await Promise.all([
                loadMetrics(),
                loadRecentEvents()
            ]);
            break;
        case 'events':
            await loadEvents();
            break;
        case 'sessions':
            await loadSessions();
            break;
        case 'tools':
            await loadToolStats();
            break;
        case 'tokens':
            await loadTokenUsage();
            break;
    }

    updateLastUpdate();
}

// Load metrics
async function loadMetrics() {
    const data = await fetchAPI('/metrics');
    if (!data) {
        updateConnectionStatus(false);
        return;
    }

    document.getElementById('tool-calls-total').textContent = data.tool_calls_total || 0;
    document.getElementById('tool-calls-success').textContent = data.tool_calls_success || 0;
    document.getElementById('tool-calls-error').textContent = data.tool_calls_error || 0;

    document.getElementById('active-sessions').textContent = data.active_sessions || 0;

    const errorRate = (data.error_rate * 100).toFixed(1);
    document.getElementById('error-rate').textContent = `${errorRate}%`;
    document.getElementById('avg-latency').textContent = `${(data.avg_latency_seconds * 1000).toFixed(0)}ms`;
}

// Load recent events for dashboard
async function loadRecentEvents() {
    const data = await fetchAPI('/events?limit=10');
    if (!data) return;

    const container = document.getElementById('recent-events-list');
    container.innerHTML = '';

    if (data.events.length === 0) {
        container.innerHTML = '<div class="event-item">No events yet</div>';
        return;
    }

    data.events.forEach(event => {
        const div = document.createElement('div');
        div.className = 'event-item';

        const time = formatTime(event.timestamp);
        const severityClass = `severity-${event.severity}`;

        div.innerHTML = `
            <div class="event-header">
                <span class="event-type">${event.event_type}</span>
                <span class="event-time">${time}</span>
            </div>
            <div class="event-details ${severityClass}">
                ${event.tool_name ? `Tool: ${event.tool_name}` : ''}
                ${event.data.message || ''}
            </div>
        `;

        container.appendChild(div);
    });
}

// Load events table
async function loadEvents() {
    const data = await fetchAPI('/events?limit=100');
    if (!data) return;

    const tbody = document.getElementById('events-table-body');
    tbody.innerHTML = '';

    data.events.forEach(event => {
        const row = document.createElement('tr');
        const time = formatTime(event.timestamp);
        const severityClass = `severity-${event.severity}`;

        row.innerHTML = `
            <td>${time}</td>
            <td>${event.event_type}</td>
            <td class="${severityClass}">${event.severity}</td>
            <td>${event.tool_name || '-'}</td>
        `;

        tbody.appendChild(row);
    });
}

// Load sessions
async function loadSessions() {
    const data = await fetchAPI('/sessions?limit=50');
    if (!data) return;

    const tbody = document.getElementById('sessions-table-body');
    tbody.innerHTML = '';

    data.sessions.forEach(session => {
        const row = document.createElement('tr');
        const created = formatTime(session.created_at);

        row.innerHTML = `
            <td><code>${session.id.substring(0, 8)}...</code></td>
            <td>${created}</td>
            <td>${session.message_count}</td>
            <td>${session.provider}</td>
            <td>${session.model}</td>
        `;

        tbody.appendChild(row);
    });
}

// Load tool statistics
async function loadToolStats() {
    const data = await fetchAPI('/tools/stats');
    if (!data) return;

    const tbody = document.getElementById('tools-table-body');
    tbody.innerHTML = '';

    data.forEach(tool => {
        const row = document.createElement('tr');
        const successRate = ((tool.successful_calls / tool.total_calls) * 100).toFixed(1);
        const lastCalled = tool.last_called ? formatTime(tool.last_called) : 'Never';

        row.innerHTML = `
            <td><code>${tool.tool_name}</code></td>
            <td>${tool.total_calls}</td>
            <td>${successRate}%</td>
            <td>${tool.avg_duration_ms.toFixed(0)}ms</td>
            <td>${lastCalled}</td>
        `;

        tbody.appendChild(row);
    });
}

// Load token usage
async function loadTokenUsage() {
    const data = await fetchAPI('/tokens/usage');
    if (!data) return;

    // Update total tokens display
    document.getElementById('tokens-total').textContent = formatNumber(data.total_tokens);
    document.getElementById('tokens-prompt').textContent = formatNumber(data.prompt_tokens);
    document.getElementById('tokens-completion').textContent = formatNumber(data.completion_tokens);

    // Render by session chart
    renderTokenChart('tokens-by-session', data.by_session, data.total_tokens);

    // Render by model chart
    renderTokenChart('tokens-by-model', data.by_model, data.total_tokens);
}

function renderTokenChart(containerId, data, total) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);

    entries.forEach(([label, amount]) => {
        const percentage = (amount / total * 100).toFixed(1);

        const div = document.createElement('div');
        div.className = 'token-bar';
        div.innerHTML = `
            <div class="token-label">${label}</div>
            <div class="token-bar-bg">
                <div class="token-bar-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="token-amount">${formatNumber(amount)} (${percentage}%)</div>
        `;

        container.appendChild(div);
    });
}

// Utility functions
function formatTime(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) { // Less than 1 minute
        return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
        const minutes = Math.floor(diff / 60000);
        return `${minutes}m ago`;
    } else if (diff < 86400000) { // Less than 1 day
        const hours = Math.floor(diff / 3600000);
        return `${hours}h ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function formatNumber(num) {
    if (num >= 1000000) {
        return `${(num / 1000000).toFixed(1)}M`;
    } else if (num >= 1000) {
        return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
}

function updateConnectionStatus(connected) {
    const status = document.getElementById('connection-status');
    if (connected) {
        status.className = 'status connected';
        status.textContent = '● Connected';
    } else {
        status.className = 'status disconnected';
        status.textContent = '● Disconnected';
    }
}

function updateLastUpdate() {
    const now = new Date();
    const time = now.toLocaleTimeString();
    document.getElementById('last-update').textContent = `Last update: ${time}`;
}

// Auto-refresh
function startPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
    }

    pollTimer = setInterval(() => {
        refreshData();
    }, POLL_INTERVAL);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    refreshData();
    startPolling();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    stopPolling();
});
