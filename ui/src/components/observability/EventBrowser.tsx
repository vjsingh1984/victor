/**
 * Event Browser - Searchable and filterable event list
 *
 * Features:
 * - Event search and filtering
 * - Multi-column sorting
 * - Pagination
 * - Event detail modal
 */

import { useState, useEffect, useCallback } from 'react';

interface EventBrowserProps {
  apiUrl?: string;
}

interface Event {
  id: string;
  event_type: string;
  timestamp: string;
  session_id: string;
  data: Record<string, unknown>;
  tool_name?: string;
  severity?: string;
}

interface EventFilters {
  eventTypes: string[];
  sessionId?: string;
  toolName?: string;
  severity?: string;
  searchQuery?: string;
}

export function EventBrowser({ apiUrl = '/obs' }: EventBrowserProps) {
  const [events, setEvents] = useState<Event[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);

  const [filters, setFilters] = useState<EventFilters>({
    eventTypes: [],
  });

  const fetchEvents = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        limit: '100',
        offset: String(page * 100),
        ...(filters.eventTypes.length > 0 && {
          event_types: filters.eventTypes.join(','),
        }),
        ...(filters.sessionId && { session_id: filters.sessionId }),
        ...(filters.toolName && { tool_name: filters.toolName }),
        ...(filters.severity && { severity: filters.severity }),
      });

      const response = await fetch(`${apiUrl}/events/recent?${params}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setEvents(data.events || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch events');
    } finally {
      setLoading(false);
    }
  }, [apiUrl, filters, page]);

  // Fetch events on mount and when filters/page change
  useEffect(() => {
    fetchEvents();
  }, [fetchEvents]);

  const handleSearch = (searchQuery: string) => {
    setFilters({ ...filters, searchQuery });
    setPage(0);
  };

  const handleFilterChange = (key: keyof EventFilters, value: unknown) => {
    setFilters({ ...filters, [key]: value });
    setPage(0);
  };

  return (
    <div className="event-browser">
      {/* Header */}
      <div className="browser-header">
        <h2>Event Browser</h2>
        <SearchBar onSearch={handleSearch} />
      </div>

      {/* Filters */}
      <EventFilters
        filters={filters}
        onChange={handleFilterChange}
      />

      {/* Error State */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="loading-state">Loading events...</div>
      )}

      {/* Events List */}
      {!loading && (
        <div className="events-list">
          {events.length === 0 ? (
            <div className="empty-state">No events found</div>
          ) : (
            events.map((event) => (
              <EventRow
                key={event.id}
                event={event}
                onClick={() => setSelectedEvent(event)}
              />
            ))
          )}
        </div>
      )}

      {/* Pagination */}
      {!loading && events.length > 0 && (
        <div className="pagination">
          <button
            onClick={() => setPage(Math.max(0, page - 1))}
            disabled={page === 0}
          >
            Previous
          </button>
          <span className="page-info">Page {page + 1}</span>
          <button
            onClick={() => setPage(page + 1)}
            disabled={events.length < 100}
          >
            Next
          </button>
        </div>
      )}

      {/* Event Detail Modal */}
      {selectedEvent && (
        <EventDetailModal
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
    </div>
  );
}

interface SearchBarProps {
  onSearch: (query: string) => void;
}

function SearchBar({ onSearch }: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(query);
  };

  return (
    <form onSubmit={handleSubmit} className="search-bar">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search events..."
        className="search-input"
      />
      <button type="submit" className="search-button">
        Search
      </button>
    </form>
  );
}

interface EventFiltersProps {
  filters: EventFilters;
  onChange: (key: keyof EventFilters, value: unknown) => void;
}

function EventFilters({ filters, onChange }: EventFiltersProps) {
  return (
    <div className="event-filters">
      <select
        value={filters.severity || ''}
        onChange={(e) => onChange('severity', e.target.value)}
        className="filter-select"
      >
        <option value="">All Severities</option>
        <option value="error">Errors</option>
        <option value="warning">Warnings</option>
        <option value="info">Info</option>
      </select>

      <input
        type="text"
        placeholder="Filter by session ID..."
        value={filters.sessionId || ''}
        onChange={(e) => onChange('sessionId', e.target.value)}
        className="filter-input"
      />

      <input
        type="text"
        placeholder="Filter by tool name..."
        value={filters.toolName || ''}
        onChange={(e) => onChange('toolName', e.target.value)}
        className="filter-input"
      />
    </div>
  );
}

interface EventRowProps {
  event: Event;
  onClick: () => void;
}

function EventRow({ event, onClick }: EventRowProps) {
  const getSeverityClass = (severity?: string) => {
    switch (severity) {
      case 'error':
        return 'severity-error';
      case 'warning':
        return 'severity-warning';
      default:
        return 'severity-info';
    }
  };

  return (
    <div
      className={`event-row ${getSeverityClass(event.severity)}`}
      onClick={onClick}
    >
      <div className="event-type">{event.event_type}</div>
      <div className="event-time">
        {new Date(event.timestamp).toLocaleString()}
      </div>
      <div className="event-session">
        {event.session_id.slice(0, 8)}...
      </div>
      {event.tool_name && (
        <div className="event-tool">{event.tool_name}</div>
      )}
    </div>
  );
}

interface EventDetailModalProps {
  event: Event;
  onClose: () => void;
}

function EventDetailModal({ event, onClose }: EventDetailModalProps) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Event Details</h3>
          <button onClick={onClose} className="close-button">
            ×
          </button>
        </div>
        <div className="modal-body">
          <div className="detail-row">
            <span className="detail-label">ID:</span>
            <span className="detail-value">{event.id}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Type:</span>
            <span className="detail-value">{event.event_type}</span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Timestamp:</span>
            <span className="detail-value">
              {new Date(event.timestamp).toLocaleString()}
            </span>
          </div>
          <div className="detail-row">
            <span className="detail-label">Session ID:</span>
            <span className="detail-value">{event.session_id}</span>
          </div>
          {event.tool_name && (
            <div className="detail-row">
              <span className="detail-label">Tool:</span>
              <span className="detail-value">{event.tool_name}</span>
            </div>
          )}
          <div className="detail-row">
            <span className="detail-label">Data:</span>
            <pre className="detail-json">
              {JSON.stringify(event.data, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
