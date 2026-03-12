import { startTransition, useEffect, useState } from 'react'
import './App.css'

type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error'

type FollowUpSuggestion = {
  command: string
  description?: string
}

type BridgeEvent = {
  id: string
  type: string
  data: Record<string, unknown>
  timestamp: number
}

const MAX_EVENTS = 40

function getDefaultWsUrl(): string {
  const configured = import.meta.env.VITE_VICTOR_WS_URL
  if (configured) {
    return configured
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws/events`
}

function isBridgeEvent(payload: unknown): payload is BridgeEvent {
  if (!payload || typeof payload !== 'object') {
    return false
  }

  const candidate = payload as Partial<BridgeEvent>
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.type === 'string' &&
    typeof candidate.timestamp === 'number' &&
    !!candidate.data &&
    typeof candidate.data === 'object'
  )
}

function extractSuggestions(event: BridgeEvent): FollowUpSuggestion[] {
  const raw = event.data.follow_up_suggestions
  if (!Array.isArray(raw)) {
    return []
  }

  return raw.filter((item): item is FollowUpSuggestion => {
    return (
      !!item &&
      typeof item === 'object' &&
      typeof (item as FollowUpSuggestion).command === 'string' &&
      (item as FollowUpSuggestion).command.trim().length > 0
    )
  })
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function formatEventTitle(event: BridgeEvent): string {
  const toolName =
    typeof event.data.tool_name === 'string'
      ? event.data.tool_name
      : typeof event.data.name === 'string'
        ? event.data.name
        : 'tool event'

  if (event.type === 'tool.start') {
    return `${toolName} started`
  }
  if (event.type === 'tool.error') {
    return `${toolName} failed`
  }
  if (event.type === 'tool.complete') {
    return `${toolName} completed`
  }
  return `${toolName} updated`
}

function formatEventDetail(event: BridgeEvent): string {
  const preview = event.data.preview
  if (preview && typeof preview === 'object') {
    const previewType = (preview as { type?: unknown }).type
    if (typeof previewType === 'string') {
      return `Preview: ${previewType}`
    }
  }

  if (typeof event.data.result_excerpt === 'string' && event.data.result_excerpt.trim()) {
    return event.data.result_excerpt
  }

  if (typeof event.data.error === 'string' && event.data.error.trim()) {
    return event.data.error
  }

  if (typeof event.data.tool_name === 'string') {
    return `Live ${event.type} event from Victor`
  }

  return 'Streaming event received from /ws/events'
}

async function copyToClipboard(text: string): Promise<boolean> {
  if (!navigator.clipboard?.writeText) {
    return false
  }

  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

function App() {
  const [events, setEvents] = useState<BridgeEvent[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const [statusMessage, setStatusMessage] = useState('Connecting to Victor event stream...')
  const [reconnectToken, setReconnectToken] = useState(0)
  const [preparedCommand, setPreparedCommand] = useState('')
  const [preparedLabel, setPreparedLabel] = useState('No follow-up selected yet')
  const [copied, setCopied] = useState(false)
  const endpoint = getDefaultWsUrl()

  useEffect(() => {
    const socket = new WebSocket(endpoint)
    setConnectionState('connecting')
    setStatusMessage('Connecting to Victor event stream...')

    socket.onopen = () => {
      setConnectionState('connected')
      setStatusMessage('Watching /ws/events for live tool activity')
      socket.send(JSON.stringify({ type: 'subscribe', categories: ['all'] }))
    }

    socket.onmessage = (messageEvent) => {
      try {
        const payload = JSON.parse(messageEvent.data)

        if (isBridgeEvent(payload)) {
          startTransition(() => {
            setEvents((current) => [payload, ...current].slice(0, MAX_EVENTS))
          })
          return
        }

        if (payload?.type === 'subscribed') {
          setStatusMessage('Subscribed to Victor event categories')
        }
      } catch {
        setStatusMessage('Received a non-JSON event payload')
      }
    }

    socket.onerror = () => {
      setConnectionState('error')
      setStatusMessage('WebSocket error while reading /ws/events')
    }

    socket.onclose = () => {
      setConnectionState('disconnected')
      setStatusMessage('Disconnected from Victor event stream')
    }

    return () => {
      socket.close()
    }
  }, [endpoint, reconnectToken])

  const toolEvents = events.filter((event) => event.type.startsWith('tool.'))
  const toolStarts = toolEvents.filter((event) => event.type === 'tool.start').length
  const toolCompletes = toolEvents.filter((event) => event.type === 'tool.complete').length
  const toolErrors = toolEvents.filter((event) => event.type === 'tool.error').length

  const handleSuggestion = async (suggestion: FollowUpSuggestion) => {
    setPreparedCommand(suggestion.command)
    setPreparedLabel(suggestion.description || 'Prepared graph follow-up')
    const wasCopied = await copyToClipboard(suggestion.command)
    setCopied(wasCopied)
  }

  const handleCopyPrepared = async () => {
    if (!preparedCommand) {
      return
    }
    const wasCopied = await copyToClipboard(preparedCommand)
    setCopied(wasCopied)
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Victor + ProximaDB</p>
          <h1>Live Tool Event Console</h1>
          <p className="lede">
            This page listens to <code>/ws/events</code>, shows live tool activity, and stages
            graph follow-ups the moment Victor suggests them.
          </p>
        </div>

        <div className="hero-panel">
          <span className={`status-pill ${connectionState}`}>{connectionState}</span>
          <p>{statusMessage}</p>
          <code>{endpoint}</code>
          <button
            className="secondary-button"
            onClick={() => setReconnectToken((value) => value + 1)}
            type="button"
          >
            Reconnect
          </button>
        </div>
      </header>

      <main className="dashboard">
        <section className="feed-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Realtime feed</p>
              <h2>Tool events</h2>
            </div>
            <div className="metric-strip">
              <span>{toolStarts} starts</span>
              <span>{toolCompletes} completions</span>
              <span>{toolErrors} errors</span>
            </div>
          </div>

          {toolEvents.length === 0 ? (
            <div className="empty-state">
              <p>No tool events yet.</p>
              <p>Start Victor, connect the API server, and trigger a tool call.</p>
            </div>
          ) : (
            <div className="event-list">
              {toolEvents.map((event) => {
                const suggestions = extractSuggestions(event)

                return (
                  <article className={`event-card ${event.type.replace('.', '-')}`} key={event.id}>
                    <div className="event-topline">
                      <span className="event-type">{event.type}</span>
                      <span className="event-time">{formatTimestamp(event.timestamp)}</span>
                    </div>

                    <h3>{formatEventTitle(event)}</h3>
                    <p className="event-detail">{formatEventDetail(event)}</p>

                    {suggestions.length > 0 ? (
                      <div className="suggestion-cluster">
                        <p className="suggestion-title">Suggested next actions</p>
                        <div className="suggestion-buttons">
                          {suggestions.slice(0, 3).map((suggestion) => (
                            <button
                              key={`${event.id}-${suggestion.command}`}
                              className="suggestion-button"
                              onClick={() => void handleSuggestion(suggestion)}
                              type="button"
                            >
                              {suggestion.description || suggestion.command}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </article>
                )
              })}
            </div>
          )}
        </section>

        <aside className="side-panel">
          <section className="prepared-panel">
            <div className="panel-header stacked">
              <div>
                <p className="section-kicker">Prepared follow-up</p>
                <h2>Graph command tray</h2>
              </div>
              <span className={`copy-indicator ${copied ? 'ready' : ''}`}>
                {copied ? 'Copied' : 'Click a suggestion to stage it'}
              </span>
            </div>

            {preparedCommand ? (
              <>
                <p className="prepared-label">{preparedLabel}</p>
                <pre>{preparedCommand}</pre>
                <div className="tray-actions">
                  <button className="primary-button" onClick={() => void handleCopyPrepared()} type="button">
                    Copy command
                  </button>
                  <button
                    className="secondary-button"
                    onClick={() => {
                      setPreparedCommand('')
                      setPreparedLabel('No follow-up selected yet')
                      setCopied(false)
                    }}
                    type="button"
                  >
                    Clear
                  </button>
                </div>
              </>
            ) : (
              <div className="empty-tray">
                <p>Follow-up suggestions from `code_search` or graph-aware tool results appear here.</p>
              </div>
            )}
          </section>

          <section className="summary-panel">
            <p className="section-kicker">Current focus</p>
            <h2>What this page is for</h2>
            <ul>
              <li>Watch live `tool.start`, `tool.complete`, and `tool.error` events.</li>
              <li>See `follow_up_suggestions` as buttons instead of buried JSON.</li>
              <li>Copy the suggested `graph(...)` command into your next workflow.</li>
            </ul>
          </section>
        </aside>
      </main>
    </div>
  )
}

export default App
