import React, { useEffect, useState } from 'react';
import { Hash } from 'lucide-react';
import { TeamExecutionView } from './components/TeamExecutionView';
import { useDashboardStore } from './store/dashboardStore';

/**
 * App component
 *
 * Main application component that handles execution selection
 * and renders the team execution view.
 */
function App() {
  const [executionId, setExecutionId] = useState<string | null>(null);
  const [inputId, setInputId] = useState('');
  const executions = useDashboardStore((state) => state.executionSummaries);
  const setExecutionSummaries = useDashboardStore((state) => state.setExecutionSummaries);
  const setLoading = useDashboardStore((state) => state.setLoading);
  const setError = useDashboardStore((state) => state.setError);

  // Load execution summaries on mount
  useEffect(() => {
    async function loadExecutions() {
      setLoading(true);
      try {
        const response = await fetch('/api/v1/executions?limit=100');
        if (response.ok) {
          const data = await response.json();
          setExecutionSummaries(data);
        }
      } catch (err) {
        console.error('Failed to load executions:', err);
        setError('Failed to load executions');
      } finally {
        setLoading(false);
      }
    }

    loadExecutions();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputId.trim()) {
      setExecutionId(inputId.trim());
    }
  };

  const handleSelectExecution = (id: string) => {
    setExecutionId(id);
    setInputId(id);
  };

  if (!executionId) {
    return (
      <div className="min-h-screen bg-gray-50 py-12 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-3 mb-4">
              <Hash className="w-12 h-12 text-blue-600" />
              <h1 className="text-4xl font-bold text-gray-900">Victor Team Dashboard</h1>
            </div>
            <p className="text-lg text-gray-600">
              Real-time collaboration dashboard for multi-agent team execution
            </p>
          </div>

          {/* Input form */}
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <input
                type="text"
                value={inputId}
                onChange={(e) => setInputId(e.target.value)}
                placeholder="Enter execution ID..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
              >
                View Execution
              </button>
            </form>
          </div>

          {/* Execution list */}
          {executions.length > 0 && (
            <div className="bg-white rounded-lg shadow overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">
                  Recent Executions ({executions.length})
                </h2>
              </div>
              <div className="divide-y divide-gray-200">
                {executions.slice(0, 10).map((execution) => (
                  <button
                    key={execution.execution_id}
                    onClick={() => handleSelectExecution(execution.execution_id)}
                    className="w-full px-6 py-4 text-left hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-medium text-gray-900">{execution.team_id}</h3>
                        <p className="text-sm text-gray-500 mt-1">
                          ID: {execution.execution_id}
                        </p>
                      </div>
                      <div className="text-right">
                        <span
                          className={`inline-block px-2 py-1 text-xs font-medium rounded-full ${
                            execution.status === 'running'
                              ? 'bg-blue-100 text-blue-800'
                              : execution.status === 'completed'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}
                        >
                          {execution.status}
                        </span>
                        <p className="text-xs text-gray-500 mt-1">
                          {execution.member_count} members
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {executions.length === 0 && (
            <div className="bg-white rounded-lg shadow p-12 text-center">
              <Hash className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No executions found</h3>
              <p className="text-gray-500">
                Start a team execution to see it appear in the dashboard
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Back button */}
        <button
          onClick={() => setExecutionId(null)}
          className="mb-6 text-blue-600 hover:text-blue-800 font-medium"
        >
          ← Back to executions
        </button>

        {/* Execution view */}
        <TeamExecutionView executionId={executionId} />
      </div>
    </div>
  );
}

export default App;
