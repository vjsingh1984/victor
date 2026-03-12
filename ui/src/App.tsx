import { startTransition, useEffect, useState, type KeyboardEvent } from 'react'
import './App.css'

type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error'

type FollowUpSuggestion = {
  command: string
  description?: string
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
}

type ChatResponsePayload = {
  content?: string
  detail?: string
  error?: string
  tool_calls?: unknown[]
}

type StreamEventPayload = {
  content?: string
  message?: string
  request_id?: string
  tool_call?: unknown
  type?: string
}

type BridgeEvent = {
  id: string
  type: string
  data: Record<string, unknown>
  timestamp: number
}

type RequestTimelineHealth = {
  liveConnected: boolean
  liveEventCount: number
  lastLiveEventAt: number | null
  lastSnapshotAt: number | null
  snapshotCount: number
  snapshotLoaded: boolean
}

const MAX_EVENTS = 40
const REQUEST_EVENT_LIMIT = 12
const EMPTY_REQUEST_TIMELINE_HEALTH: RequestTimelineHealth = {
  liveConnected: false,
  liveEventCount: 0,
  lastLiveEventAt: null,
  lastSnapshotAt: null,
  snapshotCount: 0,
  snapshotLoaded: false,
}

function buildEventSubscription(
  categories: string[],
  correlationId?: string | null,
): string {
  return JSON.stringify({
    type: 'subscribe',
    categories,
    ...(correlationId ? { correlation_id: correlationId } : {}),
  })
}

function trimTrailingSlash(value: string): string {
  return value.endsWith('/') ? value.slice(0, -1) : value
}

function getDefaultApiBase(): string {
  const configured = import.meta.env.VITE_VICTOR_API_URL
  if (configured) {
    return trimTrailingSlash(configured)
  }

  const isViteDevServer =
    window.location.hostname === 'localhost' &&
    ['5173', '4173'].includes(window.location.port)

  if (isViteDevServer) {
    return 'http://127.0.0.1:8765'
  }

  return trimTrailingSlash(window.location.origin)
}

function getDefaultWsUrl(apiBase: string): string {
  const configured = import.meta.env.VITE_VICTOR_WS_URL
  if (configured) {
    return configured
  }

  const target = new URL(apiBase)
  target.protocol = target.protocol === 'https:' ? 'wss:' : 'ws:'
  target.pathname = '/ws/events'
  target.search = ''
  target.hash = ''
  return target.toString()
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

function getEventCorrelationId(event: BridgeEvent): string | null {
  const requestId = event.data.request_id
  if (typeof requestId === 'string' && requestId.trim()) {
    return requestId
  }

  const correlationId = event.data.correlation_id
  if (typeof correlationId === 'string' && correlationId.trim()) {
    return correlationId
  }

  return null
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

async function readChatResponse(response: Response): Promise<ChatResponsePayload> {
  const contentType = response.headers.get('content-type') || ''

  if (contentType.includes('application/json')) {
    return (await response.json()) as ChatResponsePayload
  }

  const text = await response.text()
  if (!text.trim()) {
    return {}
  }

  try {
    return JSON.parse(text) as ChatResponsePayload
  } catch {
    return response.ok ? { content: text } : { error: text }
  }
}

function formatChatReply(payload: ChatResponsePayload): string {
  let reply = payload.content?.trim() || ''
  if (!reply && Array.isArray(payload.tool_calls) && payload.tool_calls.length > 0) {
    reply = describeToolCalls(payload.tool_calls)
  }
  if (!reply) {
    reply = 'Victor returned an empty response.'
  }
  return reply
}

function mergeEvents(
  current: BridgeEvent[],
  incoming: BridgeEvent[],
  limit: number,
): BridgeEvent[] {
  const merged: BridgeEvent[] = []
  const seen = new Set<string>()

  for (const event of [...incoming, ...current]) {
    if (seen.has(event.id)) {
      continue
    }
    seen.add(event.id)
    merged.push(event)
    if (merged.length >= limit) {
      break
    }
  }

  return merged
}

function formatRelativeAge(timestampMs: number | null, nowMs: number): string | null {
  if (!timestampMs) {
    return null
  }

  const deltaSeconds = Math.max(0, Math.round((nowMs - timestampMs) / 1000))
  if (deltaSeconds < 60) {
    return `${deltaSeconds}s ago`
  }

  const deltaMinutes = Math.round(deltaSeconds / 60)
  if (deltaMinutes < 60) {
    return `${deltaMinutes}m ago`
  }

  const deltaHours = Math.round(deltaMinutes / 60)
  return `${deltaHours}h ago`
}

function formatWallClock(timestampMs: number | null): string | null {
  if (!timestampMs) {
    return null
  }

  return new Date(timestampMs).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function getTimelineActivityState(
  requestId: string | null,
  health: RequestTimelineHealth,
  nowMs: number,
): string {
  if (!requestId) {
    return 'waiting'
  }
  if (!health.snapshotLoaded && !health.liveConnected && health.liveEventCount === 0) {
    return 'warming'
  }

  const liveAgeMs = health.lastLiveEventAt ? nowMs - health.lastLiveEventAt : null
  if (health.liveConnected && liveAgeMs !== null && liveAgeMs <= 15000) {
    return 'live active'
  }
  if (health.liveConnected && (health.snapshotLoaded || health.liveEventCount > 0)) {
    return 'live idle'
  }
  if (!health.liveConnected && liveAgeMs !== null && liveAgeMs > 45000) {
    return 'stale'
  }
  if (health.snapshotCount > 0 && health.liveEventCount > 0) {
    return 'snapshot + live'
  }
  if (health.liveEventCount > 0) {
    return 'live only'
  }
  if (health.snapshotLoaded && health.snapshotCount > 0) {
    return 'snapshot only'
  }
  if (health.snapshotLoaded) {
    return 'no events yet'
  }
  return 'warming'
}

function formatRequestTimelineMode(
  requestId: string | null,
  health: RequestTimelineHealth,
  nowMs: number,
): string {
  return getTimelineActivityState(requestId, health, nowMs)
}

function formatRequestTimelineDetail(
  requestId: string | null,
  health: RequestTimelineHealth,
  nowMs: number,
): string {
  if (!requestId) {
    return 'No active request timeline yet.'
  }

  const parts: string[] = []
  if (health.snapshotLoaded) {
    parts.push(health.snapshotCount > 0 ? `backfilled ${health.snapshotCount}` : 'snapshot checked')
  }
  if (health.liveConnected) {
    parts.push(health.liveEventCount > 0 ? `live ${health.liveEventCount}` : 'live connected')
  } else if (health.liveEventCount > 0) {
    parts.push(`received ${health.liveEventCount} live event(s)`)
  }

  const activityState = getTimelineActivityState(requestId, health, nowMs)
  if (activityState === 'stale') {
    parts.push('live delivery looks stale')
  } else if (activityState === 'live idle') {
    parts.push('timeline is healthy but idle')
  }

  return parts.join(' · ') || 'Opening snapshot and live request channels.'
}

function formatRequestTimelineTimestamps(
  requestId: string | null,
  health: RequestTimelineHealth,
  nowMs: number,
): string {
  if (!requestId) {
    return 'Waiting for a request ID from Victor.'
  }

  const parts: string[] = []
  const snapshotAge = formatRelativeAge(health.lastSnapshotAt, nowMs)
  const snapshotClock = formatWallClock(health.lastSnapshotAt)
  if (snapshotAge && snapshotClock) {
    parts.push(`snapshot ${snapshotClock} (${snapshotAge})`)
  }

  const liveAge = formatRelativeAge(health.lastLiveEventAt, nowMs)
  const liveClock = formatWallClock(health.lastLiveEventAt)
  if (liveAge && liveClock) {
    parts.push(`live ${liveClock} (${liveAge})`)
  } else if (health.liveConnected) {
    parts.push('live socket connected')
  }

  return parts.join(' · ') || 'Request timeline channels are still opening.'
}

function timelineModeClassName(mode: string): string {
  return mode.toLowerCase().replace(/[^a-z0-9]+/g, '-')
}

function extractSsePayloads(buffer: string): { payloads: string[]; remainder: string } {
  const normalized = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  const segments = normalized.split('\n\n')
  const remainder = segments.pop() || ''
  const payloads = segments
    .map((segment) =>
      segment
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trimStart())
        .join('\n'),
    )
    .filter((payload) => payload.length > 0)

  return { payloads, remainder }
}

function describeToolCalls(payload: unknown): string {
  const calls = Array.isArray(payload) ? payload : payload ? [payload] : []
  if (calls.length === 0) {
    return 'Victor requested a tool call.'
  }

  const labels = calls.map((call) => {
    if (!call || typeof call !== 'object') {
      return 'tool call'
    }

    const record = call as Record<string, unknown>
    if (typeof record.name === 'string' && record.name.trim()) {
      return record.name
    }
    if (typeof record.tool_name === 'string' && record.tool_name.trim()) {
      return record.tool_name
    }

    const nestedFunction = record.function
    if (nestedFunction && typeof nestedFunction === 'object') {
      const functionName = (nestedFunction as Record<string, unknown>).name
      if (typeof functionName === 'string' && functionName.trim()) {
        return functionName
      }
    }

    return 'tool call'
  })

  const uniqueLabels = [...new Set(labels)]
  const summary = uniqueLabels.join(', ')
  return uniqueLabels.length === 1
    ? `Victor requested tool call: ${summary}.`
    : `Victor requested tool calls: ${summary}.`
}

function buildStreamReply(content: string, notes: string[]): string {
  const blocks = [content.trim(), notes.join('\n\n').trim()].filter(Boolean)
  if (blocks.length === 0) {
    return 'Streaming response...'
  }
  return blocks.join('\n\n')
}

function App() {
  const [events, setEvents] = useState<BridgeEvent[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const [statusMessage, setStatusMessage] = useState('Connecting to Victor event stream...')
  const [reconnectToken, setReconnectToken] = useState(0)
  const [preparedCommand, setPreparedCommand] = useState('')
  const [preparedLabel, setPreparedLabel] = useState('No follow-up selected yet')
  const [draft, setDraft] = useState('')
  const [activeRequestId, setActiveRequestId] = useState<string | null>(null)
  const [requestEvents, setRequestEvents] = useState<BridgeEvent[]>([])
  const [requestTimelineHealth, setRequestTimelineHealth] = useState<RequestTimelineHealth>(
    EMPTY_REQUEST_TIMELINE_HEALTH,
  )
  const [requestTimelineNow, setRequestTimelineNow] = useState(() => Date.now())
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [copied, setCopied] = useState(false)
  const apiBase = getDefaultApiBase()
  const endpoint = getDefaultWsUrl(apiBase)

  useEffect(() => {
    const socket = new WebSocket(endpoint)
    setConnectionState('connecting')
    setStatusMessage('Connecting to Victor event stream...')

    socket.onopen = () => {
      setConnectionState('connected')
      setStatusMessage('Watching /ws/events for live tool activity')
      socket.send(buildEventSubscription(['all']))
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

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setRequestTimelineNow(Date.now())
    }, 15000)

    return () => {
      window.clearInterval(intervalId)
    }
  }, [])

  useEffect(() => {
    setRequestTimelineNow(Date.now())
    setRequestEvents([])
    setRequestTimelineHealth({ ...EMPTY_REQUEST_TIMELINE_HEALTH })

    if (!activeRequestId) {
      return
    }

    const controller = new AbortController()
    const socket = new WebSocket(endpoint)
    const categories = ['tool.start', 'tool.progress', 'tool.complete', 'tool.error']

    socket.onopen = () => {
      setRequestTimelineHealth((current) => ({ ...current, liveConnected: true }))
      socket.send(buildEventSubscription(categories, activeRequestId))
    }

    socket.onmessage = (messageEvent) => {
      try {
        const payload = JSON.parse(messageEvent.data)
        if (!isBridgeEvent(payload)) {
          return
        }

        startTransition(() => {
          setRequestEvents((current) => mergeEvents(current, [payload], REQUEST_EVENT_LIMIT))
        })
        setRequestTimelineNow(Date.now())
        setRequestTimelineHealth((current) => ({
          ...current,
          liveConnected: true,
          liveEventCount: current.liveEventCount + 1,
          lastLiveEventAt: Date.now(),
        }))
      } catch {
        return
      }
    }

    socket.onclose = () => {
      setRequestTimelineHealth((current) => ({ ...current, liveConnected: false }))
    }

    const hydrateRecentEvents = async () => {
      try {
        const params = new URLSearchParams({
          limit: String(REQUEST_EVENT_LIMIT),
          correlation_id: activeRequestId,
        })
        for (const category of categories) {
          params.append('categories', category)
        }

        const response = await fetch(`${apiBase}/events/recent?${params.toString()}`, {
          signal: controller.signal,
        })
        if (!response.ok) {
          return
        }

        const payload = (await response.json()) as { events?: unknown[] }
        if (!Array.isArray(payload.events)) {
          setRequestTimelineHealth((current) => ({ ...current, snapshotLoaded: true }))
          return
        }

        const snapshot = payload.events.filter(isBridgeEvent)
        startTransition(() => {
          setRequestEvents((current) => mergeEvents(current, snapshot, REQUEST_EVENT_LIMIT))
        })
        setRequestTimelineNow(Date.now())
        setRequestTimelineHealth((current) => ({
          ...current,
          lastSnapshotAt: Date.now(),
          snapshotLoaded: true,
          snapshotCount: snapshot.length,
        }))
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }
        setRequestTimelineHealth((current) => ({ ...current, snapshotLoaded: true }))
      }
    }

    void hydrateRecentEvents()

    return () => {
      controller.abort()
      socket.close()
    }
  }, [activeRequestId, apiBase, endpoint])

  const toolEvents = events.filter((event) => event.type.startsWith('tool.'))
  const toolStarts = toolEvents.filter((event) => event.type === 'tool.start').length
  const toolCompletes = toolEvents.filter((event) => event.type === 'tool.complete').length
  const toolErrors = toolEvents.filter((event) => event.type === 'tool.error').length
  const timelineMode = formatRequestTimelineMode(
    activeRequestId,
    requestTimelineHealth,
    requestTimelineNow,
  )
  const timelineDetail = formatRequestTimelineDetail(
    activeRequestId,
    requestTimelineHealth,
    requestTimelineNow,
  )
  const timelineMeta = formatRequestTimelineTimestamps(
    activeRequestId,
    requestTimelineHealth,
    requestTimelineNow,
  )

  const handleSuggestion = async (suggestion: FollowUpSuggestion) => {
    setPreparedCommand(suggestion.command)
    setPreparedLabel(suggestion.description || 'Prepared graph follow-up')
    setDraft(suggestion.command)
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

  const upsertAssistantMessage = (content: string) => {
    setChatMessages((current) => {
      const next = [...current]
      if (next.length === 0 || next[next.length - 1]?.role !== 'assistant') {
        next.push({ role: 'assistant', content })
        return next
      }

      next[next.length - 1] = { role: 'assistant', content }
      return next
    })
  }

  const handleRunPrepared = async () => {
    const message = draft.trim()
    if (!message || isSending) {
      return
    }

    setIsSending(true)
    setActiveRequestId(null)
    setRequestEvents([])
    setRequestTimelineHealth({ ...EMPTY_REQUEST_TIMELINE_HEALTH })
    setChatMessages((current) => [
      ...current,
      { role: 'user', content: message },
      { role: 'assistant', content: 'Streaming response...' },
    ])

    try {
      const response = await fetch(`${apiBase}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: message }],
        }),
      })

      if (!response.ok) {
        const payload = await readChatResponse(response)
        const errorMessage = payload.error || payload.detail
        throw new Error(errorMessage || `Request failed with ${response.status}`)
      }

      const responseRequestId = response.headers.get('X-Victor-Request-Id')
      if (responseRequestId) {
        setActiveRequestId(responseRequestId)
      }

      const contentType = response.headers.get('content-type') || ''
      if (!contentType.includes('text/event-stream')) {
        const payload = await readChatResponse(response)
        upsertAssistantMessage(formatChatReply(payload))
        setDraft('')
        return
      }

      let assistantContent = ''
      const streamNotes: string[] = []

      const applyStreamPayload = (payloadText: string): boolean => {
        if (payloadText === '[DONE]') {
          return true
        }

        let payload: StreamEventPayload
        try {
          payload = JSON.parse(payloadText) as StreamEventPayload
        } catch {
          return false
        }

        if (typeof payload.request_id === 'string' && payload.request_id.trim()) {
          setActiveRequestId(payload.request_id)
        }

        if (payload.type === 'error') {
          throw new Error(payload.message || 'Streaming request failed')
        }

        if (payload.type === 'content' && typeof payload.content === 'string') {
          assistantContent += payload.content
          upsertAssistantMessage(buildStreamReply(assistantContent, streamNotes))
          return false
        }

        if (payload.type === 'tool_call') {
          streamNotes.push(describeToolCalls(payload.tool_call))
          upsertAssistantMessage(buildStreamReply(assistantContent, streamNotes))
        }

        return false
      }

      if (!response.body) {
        const fullStream = await response.text()
        const { payloads } = extractSsePayloads(`${fullStream}\n\n`)
        for (const payloadText of payloads) {
          const shouldStop = applyStreamPayload(payloadText)
          if (shouldStop) {
            break
          }
        }
      } else {
        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let shouldStop = false

        while (!shouldStop) {
          const { done, value } = await reader.read()
          if (done) {
            buffer += decoder.decode()
            break
          }

          buffer += decoder.decode(value, { stream: true })
          const parsed = extractSsePayloads(buffer)
          buffer = parsed.remainder

          for (const payloadText of parsed.payloads) {
            shouldStop = applyStreamPayload(payloadText)
            if (shouldStop) {
              await reader.cancel()
              break
            }
          }
        }

        if (!shouldStop && buffer.trim()) {
          const parsed = extractSsePayloads(`${buffer}\n\n`)
          for (const payloadText of parsed.payloads) {
            shouldStop = applyStreamPayload(payloadText)
            if (shouldStop) {
              break
            }
          }
        }
      }

      if (!assistantContent.trim() && streamNotes.length === 0) {
        upsertAssistantMessage('Victor returned an empty response.')
      }

      setDraft('')
    } catch (error) {
      const messageText = error instanceof Error ? error.message : 'Unknown request failure'
      upsertAssistantMessage(`Request failed: ${messageText}`)
    } finally {
      setIsSending(false)
    }
  }

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      void handleRunPrepared()
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Victor + ProximaDB</p>
          <h1>Live Tool Event Console</h1>
          <p className="lede">
            This page listens to <code>/ws/events</code>, shows live tool activity, and stages
            or runs graph follow-ups the moment Victor suggests them.
          </p>
        </div>

        <div className="hero-panel">
          <span className={`status-pill ${connectionState}`}>{connectionState}</span>
          <p>{statusMessage}</p>
          <code>{endpoint}</code>
          <p className="endpoint-label">Streaming Chat API</p>
          <code>{apiBase}/chat/stream</code>
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
                const isActiveRequestEvent =
                  !!activeRequestId && getEventCorrelationId(event) === activeRequestId

                return (
                  <article
                    className={`event-card ${event.type.replace('.', '-')} ${isActiveRequestEvent ? 'matched-request' : ''}`}
                    key={event.id}
                  >
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
                      setDraft('')
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
                <p>
                  Follow-up suggestions from `code_search` or graph-aware tool results appear
                  here, ready to copy or send to Victor.
                </p>
              </div>
            )}
          </section>

          <section className="summary-panel">
            <div className="panel-header stacked">
              <div>
                <p className="section-kicker">Execution lane</p>
                <h2>Run the staged follow-up</h2>
              </div>
            </div>

            <label className="composer-label" htmlFor="prepared-command">
              Prompt draft
            </label>
            <textarea
              id="prepared-command"
              className="composer"
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="Select a suggestion or type a direct Victor prompt"
              rows={5}
              value={draft}
            />
            <p className="composer-note">Press Ctrl+Enter or Cmd+Enter to run the drafted command.</p>

            <div className="tray-actions">
              <button
                className="primary-button"
                disabled={isSending || !draft.trim()}
                onClick={() => void handleRunPrepared()}
                type="button"
              >
                {isSending ? 'Running...' : 'Run in Victor'}
              </button>
            </div>

            <div className="request-link-panel">
              <div className="request-link-header">
                <div className="request-link-copy">
                  <p className="section-kicker">Linked timeline</p>
                  <h3>Current request</h3>
                  <p className="request-health-detail">
                    {timelineDetail}
                  </p>
                  <p className="request-health-meta">{timelineMeta}</p>
                </div>
                <div className="request-pill-group">
                  <span
                    className={`timeline-pill ${activeRequestId ? 'ready' : ''} ${timelineModeClassName(timelineMode)}`}
                  >
                    {timelineMode}
                  </span>
                  <span className={`request-pill ${activeRequestId ? 'ready' : ''}`}>
                    {activeRequestId ? activeRequestId : 'waiting'}
                  </span>
                </div>
              </div>

              {requestEvents.length === 0 ? (
                <div className="empty-tray compact">
                  <p>
                    Tool events from the active streamed request will collect here once Victor
                    starts calling tools.
                  </p>
                </div>
              ) : (
                <div className="linked-event-list">
                  {requestEvents.slice(0, 4).map((event) => (
                    <article className="linked-event-card" key={`active-${event.id}`}>
                      <div className="linked-event-topline">
                        <span>{event.type}</span>
                        <span>{formatTimestamp(event.timestamp)}</span>
                      </div>
                      <strong>{formatEventTitle(event)}</strong>
                      <p>{formatEventDetail(event)}</p>
                    </article>
                  ))}
                </div>
              )}
            </div>

            <div className="chat-log">
              {chatMessages.length === 0 ? (
                <div className="empty-tray compact">
                  <p>The response log for executed follow-ups will appear here.</p>
                </div>
              ) : (
                chatMessages.slice(-4).map((message, index) => (
                  <article className={`chat-bubble ${message.role}`} key={`${message.role}-${index}`}>
                    <span>{message.role}</span>
                    <p>{message.content}</p>
                  </article>
                ))
              )}
            </div>
          </section>
        </aside>
      </main>
    </div>
  )
}

export default App
