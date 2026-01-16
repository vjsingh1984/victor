# Victor Team Dashboard

Real-time collaboration dashboard for monitoring multi-agent team execution in Victor AI.

![Team Dashboard](/assets/dashboard-screenshot.png)

## Features

- **Real-time Updates**: WebSocket-powered streaming with sub-second latency
- **Team Monitoring**: Track execution status, member states, and progress
- **Communication Flow**: Interactive diagram showing message passing between members
- **Shared Context**: Live view of team-wide key-value store
- **Negotiation Tracking**: Visual representation of voting and consensus
- **Metrics Dashboard**: Aggregated statistics and analytics

## Quick Start

```bash
# Install dependencies
npm install

# Start development servers (frontend + backend)
./run.sh

# Or start individually
npm run dev  # Frontend only (port 3000)
# Backend runs on port 8000
```

Access the dashboard at: http://localhost:3000

## Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Format code
npm run format
```

## Project Structure

```
src/
├── components/          # React components
│   ├── TeamExecutionView.tsx
│   ├── MemberStatusCard.tsx
│   ├── CommunicationFlow.tsx
│   ├── SharedContextTable.tsx
│   ├── NegotiationPanel.tsx
│   └── MetricsPanel.tsx
├── hooks/              # Custom React hooks
│   └── useWebSocket.ts
├── store/              # State management (Zustand)
│   └── dashboardStore.ts
├── types/              # TypeScript definitions
│   └── index.ts
├── App.tsx             # Root component
└── main.tsx            # Entry point
```

## Technologies

- **React 18**: UI framework
- **TypeScript**: Type safety
- **Zustand**: State management
- **ReactFlow**: Communication flow diagrams
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Styling (via CDN)
- **Lucide Icons**: Icon library

## WebSocket Integration

The dashboard connects to the Victor backend via WebSocket:

```typescript
const wsUrl = `ws://localhost:8000/ws/team/${executionId}`;
const { state } = useWebSocket(wsUrl, {
  onEvent: (event) => handleDashboardEvent(event),
  reconnectInterval: 3000,
  maxReconnectAttempts: 10,
});
```

## API Integration

REST API for fetching execution data:

```typescript
// List executions
fetch('/api/v1/executions')

// Get execution details
fetch(`/api/v1/executions/${executionId}`)

// Get metrics
fetch('/api/v1/metrics/summary')
```

## Configuration

### Backend Port

Edit `vite.config.ts` to change the proxy configuration:

```typescript
server: {
  port: 3000,
  proxy: {
    '/ws': {
      target: 'http://localhost:8000',  // Backend URL
      ws: true,
    },
    '/api': {
      target: 'http://localhost:8000',  // Backend URL
      changeOrigin: true,
    },
  },
}
```

### WebSocket URL

Pass custom WebSocket URL to components:

```typescript
<TeamExecutionView
  executionId="exec-123"
  wsUrl="wss://api.example.com/ws/team/exec-123"
/>
```

## Production Deployment

```bash
# Build frontend
npm run build

# Output is in /dist directory
# Serve with any static file server

# Example with nginx:
# location / {
#   root /path/to/tools/team_dashboard/dist;
#   try_files $uri $uri/ /index.html;
# }
```

## Troubleshooting

### WebSocket Connection Failed

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check browser console for errors
3. Ensure CORS is configured on backend

### No Data Displayed

1. Check browser network tab for API errors
2. Verify execution ID is valid
3. Check backend logs for errors

### High Memory Usage

1. Limit communication log history in backend
2. Implement pagination for execution lists
3. Clear old execution states

## Contributing

1. Follow existing code style
2. Add TypeScript types for new props
3. Use Tailwind CSS for styling
4. Test with different team formations
5. Update documentation

## License

Apache License 2.0

See [Victor main repository](https://github.com/yourusername/victor) for details.
