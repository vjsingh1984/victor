import { useState, useEffect, useRef, useCallback } from 'react';
import Message from './components/Message';
import MessageInput from './components/MessageInput';
import ThemeToggle from './components/ThemeToggle';

interface ChatMessage {
  text: string;
  sender: 'user' | 'assistant';
  kind?: 'tool' | 'normal';
  streaming?: boolean;
}

interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

const WELCOME: ChatMessage = {
  text: "Welcome! I'm Victor, your AI assistant. How can I help you design something amazing today?",
  sender: "assistant",
};

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const RECONNECT_INTERVAL = 2000; // 2 seconds
const MAX_RECONNECT_ATTEMPTS = 5;

function loadInitialSessions(): ChatSession[] {
  try {
    const saved = localStorage.getItem('chatSessions');
    if (saved) {
      const parsed: ChatSession[] = JSON.parse(saved);
      if (parsed.length) return parsed;
    }
  } catch (error) {
    console.error("Failed to parse chat sessions from localStorage", error);
  }
  return [createSession("Session 1")];
}

function createSession(title?: string): ChatSession {
  const id = crypto.randomUUID();
  return { id, title: title || "New session", messages: [WELCOME] };
}

function App() {
  const initialSessionsRef = useRef<ChatSession[] | null>(null);
  if (!initialSessionsRef.current) {
    initialSessionsRef.current = loadInitialSessions();
  }

  const [sessions, setSessions] = useState<ChatSession[]>(() => initialSessionsRef.current || []);
  const [selectedId, setSelectedId] = useState<string>(() => initialSessionsRef.current?.[0]?.id || "");

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [isTyping, setIsTyping] = useState(false);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const ws = useRef<WebSocket | null>(null);
  const wsSessionIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const messageQueueRef = useRef<string[]>([]);
  const reconnectAttemptsRef = useRef<number>(0);

  const selectedSession = sessions.find((s) => s.id === selectedId) || sessions[0];
  const messages = selectedSession ? selectedSession.messages : [];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
    try {
      localStorage.setItem('chatSessions', JSON.stringify(sessions));
    } catch (error) {
      console.error("Failed to save chat sessions to localStorage", error);
    }
  }, [sessions]);

  // Online/offline detection
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      console.log("Network: Online");
    };
    const handleOffline = () => {
      setIsOnline(false);
      setConnectionStatus('disconnected');
      console.log("Network: Offline");
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const connectWebSocket = useCallback(() => {
    if (!selectedSession || !isOnline) {
      console.log("Skipping WebSocket connection: session or network unavailable");
      return;
    }

    // Avoid duplicate connections for the same session while one is open/connecting
    if (
      ws.current &&
      wsSessionIdRef.current === selectedSession.id &&
      (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)
    ) {
      return ws.current;
    }

    // Clear existing connection when switching sessions or after failure
    if (ws.current && ws.current.readyState !== WebSocket.CLOSED) {
      ws.current.close();
    }

    setConnectionStatus('connecting');
    const socket = new WebSocket(`${WS_URL}?session_id=${selectedSession.id}`);
    ws.current = socket;
    wsSessionIdRef.current = selectedSession.id;

    socket.onopen = () => {
      console.log("WebSocket connected", selectedSession.id);
      setConnectionStatus('connected');
      reconnectAttemptsRef.current = 0;
      setReconnectAttempts(0);

      // Process queued messages
      while (messageQueueRef.current.length > 0) {
        const queuedMessage = messageQueueRef.current.shift();
        if (queuedMessage && socket.readyState === WebSocket.OPEN) {
          socket.send(queuedMessage);
        }
      }
    };

    socket.onmessage = (event) => {
      const data = String(event.data);

      // Handle special messages
      if (data.startsWith("[session]")) {
        return;
      }

      if (data.startsWith("[ping]")) {
        // Heartbeat received, connection is alive
        return;
      }

      if (data.startsWith("[error]")) {
        const errorMsg = data.replace("[error]", "").trim();
        setSessions((prev) =>
          prev.map((sess) => {
            if (!selectedSession || sess.id !== selectedSession.id) return sess;
            const messages = [...sess.messages];
            messages.push({
              text: `Error: ${errorMsg}`,
              sender: "assistant",
              kind: "normal"
            });
            return { ...sess, messages };
          })
        );
        setIsTyping(false);
        return;
      }

      const isTool = data.startsWith("[tool]");
      const isFinalChunk = data.length === 0;

      // Stop typing indicator on final chunk
      if (isFinalChunk) {
        setIsTyping(false);
      } else if (!isTool) {
        setIsTyping(true);
      }

      setSessions((prev) =>
        prev.map((sess) => {
          if (!selectedSession || sess.id !== selectedSession.id) return sess;
          const messages = [...sess.messages];

          if (isTool) {
            messages.push({ text: data, sender: "assistant", kind: "tool" });
            return { ...sess, messages };
          }

          const last = messages[messages.length - 1];
          if (last && last.sender === "assistant" && last.kind !== "tool" && last.streaming) {
            const newStreaming = isFinalChunk ? false : true;
            messages[messages.length - 1] = {
              ...last,
              text: isFinalChunk ? last.text : last.text + data,
              streaming: newStreaming
            };
            return { ...sess, messages };
          }

          if (!isFinalChunk) {
            const streaming = true;
            messages.push({ text: data, sender: "assistant", streaming });
          }

          return { ...sess, messages };
        })
      );
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setConnectionStatus('disconnected');
    };

    socket.onclose = (event) => {
      console.log("WebSocket disconnected", selectedSession.id, event.code, event.reason);
      setConnectionStatus('disconnected');
      setIsTyping(false);
      ws.current = null;
      wsSessionIdRef.current = null;

      // Attempt reconnection if not intentional close
      if (isOnline && reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS && event.code !== 1000) {
        setConnectionStatus('reconnecting');
        const delay = Math.min(RECONNECT_INTERVAL * Math.pow(2, reconnectAttemptsRef.current), 30000);
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/${MAX_RECONNECT_ATTEMPTS})`);

        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectAttemptsRef.current += 1;
          setReconnectAttempts(reconnectAttemptsRef.current);
          // Trigger reconnection by calling connectWebSocket
          connectWebSocket();
        }, delay);
      } else if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
        setSessions((prev) =>
          prev.map((sess) => {
            if (!selectedSession || sess.id !== selectedSession.id) return sess;
            return {
              ...sess,
              messages: [
                ...sess.messages,
                {
                  text: "Connection lost after multiple retry attempts. Please refresh the page.",
                  sender: "assistant",
                  kind: "normal"
                }
              ]
            };
          })
        );
      }
    };

    return socket;
  }, [selectedSession?.id, isOnline]);

  useEffect(() => {
    const socket = connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socket) {
        socket.close(1000, "Component unmounting");
        ws.current = null;
        wsSessionIdRef.current = null;
      }
    };
  }, [connectWebSocket]);

  const handleSendMessage = (text: string) => {
    if (!selectedSession) return;

    // Add message to UI immediately
    setSessions((prev) =>
      prev.map((sess) =>
        sess.id === selectedSession.id
          ? { ...sess, messages: [...sess.messages, { text, sender: "user" }] }
          : sess
      )
    );

    // Send if connected, queue if not
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(text);
      setIsTyping(true);
    } else {
      messageQueueRef.current.push(text);
      if (connectionStatus === 'disconnected') {
        setSessions((prev) =>
          prev.map((sess) =>
            sess.id === selectedSession.id
              ? {
                  ...sess,
                  messages: [
                    ...sess.messages,
                    {
                      text: "Message queued. Attempting to reconnect...",
                      sender: "assistant",
                      kind: "normal"
                    }
                  ]
                }
              : sess
          )
        );
        // Attempt reconnection
        setReconnectAttempts(0);
        connectWebSocket();
      }
    }
  };

  const handleNewSession = () => {
    const title = `Session ${sessions.length + 1}`;
    const newSession = createSession(title);
    setSessions((prev) => [newSession, ...prev]);
    setSelectedId(newSession.id);
  };

  const handleSelectSession = (id: string) => {
    setSelectedId(id);
    setReconnectAttempts(0);
  };

  const handleClearSession = () => {
    if (!selectedSession) return;
    const cleared = { ...selectedSession, messages: [WELCOME] };
    setSessions((prev) => prev.map((s) => (s.id === cleared.id ? cleared : s)));
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send("__reset_session__");
    }
  };

  // Connection status badge
  const getStatusBadge = () => {
    const statusConfig = {
      connecting: { bg: 'bg-yellow-500', text: 'Connecting...', pulse: true },
      connected: { bg: 'bg-green-500', text: 'Connected', pulse: false },
      disconnected: { bg: 'bg-red-500', text: 'Disconnected', pulse: false },
      reconnecting: { bg: 'bg-orange-500', text: 'Reconnecting...', pulse: true },
    };

    const config = statusConfig[connectionStatus];

    return (
      <div className="flex items-center gap-2 text-sm">
        <div className={`w-2 h-2 rounded-full ${config.bg} ${config.pulse ? 'animate-pulse' : ''}`} />
        <span className="text-gray-600 dark:text-gray-300">{config.text}</span>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      {/* Offline Banner */}
      {!isOnline && (
        <div className="fixed top-0 left-0 right-0 bg-red-600 text-white text-center py-2 z-50">
          You are offline. Messages will be queued until connection is restored.
        </div>
      )}

      <aside className="hidden md:flex md:w-72 lg:w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 p-3 flex-col">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-gray-800 dark:text-white">Conversations</h2>
          <button
            className="px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            onClick={handleNewSession}
          >
            +
          </button>
        </div>
        <div className="flex-1 overflow-y-auto space-y-2">
          {sessions.map((sess) => (
            <button
              key={sess.id}
              className={`w-full text-left px-3 py-2 rounded transition-colors ${
                selectedSession && sess.id === selectedSession.id
                  ? "bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100"
                  : "bg-gray-50 dark:bg-gray-700 text-gray-800 dark:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-600"
              }`}
              onClick={() => handleSelectSession(sess.id)}
            >
              {sess.title}
            </button>
          ))}
        </div>
        <button
          className="mt-3 px-3 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
          onClick={handleClearSession}
        >
          Clear Session
        </button>
      </aside>

      <div className="flex flex-col flex-1 min-w-0">
        <header className="bg-white dark:bg-gray-800 shadow-md">
          <div className="max-w-6xl w-full mx-auto py-3 px-4 flex justify-between items-center">
            <h1 className="text-xl font-bold text-gray-800 dark:text-white">Victor AI Assistant</h1>
            <div className="flex items-center gap-4">
              {getStatusBadge()}
              <ThemeToggle />
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-200 dark:scrollbar-thumb-gray-600 dark:scrollbar-track-gray-800">
          <div className="max-w-6xl w-full mx-auto p-4">
            <div className="flex flex-col space-y-4">
              {messages.map((msg, index) => (
                <Message key={`${selectedSession?.id}-${index}`} text={msg.text} sender={msg.sender} kind={msg.kind} />
              ))}
              {isTyping && (
                <div className="flex items-start justify-start w-full">
                  <div className="px-4 py-3 bg-gray-200 dark:bg-gray-700 rounded-2xl rounded-bl-none">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </main>

        <footer className="bg-white dark:bg-gray-800 shadow-t">
          <div className="max-w-6xl w-full mx-auto p-4">
            <MessageInput
              onSendMessage={handleSendMessage}
              disabled={!isOnline && connectionStatus === 'disconnected'}
            />
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
