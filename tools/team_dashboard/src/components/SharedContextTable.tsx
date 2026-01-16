import React, { useState } from 'react';
import { Search, ChevronDown, ChevronRight } from 'lucide-react';

interface SharedContextTableProps {
  context: Record<string, unknown>;
  className?: string;
}

/**
 * Value display component
 */
function ValueDisplay({ value }: { value: unknown }): JSX.Element {
  const [expanded, setExpanded] = useState(false);

  if (value === null) {
    return <span className="text-gray-400 italic">null</span>;
  }

  if (value === undefined) {
    return <span className="text-gray-400 italic">undefined</span>;
  }

  if (typeof value === 'boolean') {
    return (
      <span className={value ? 'text-green-600' : 'text-red-600'}>
        {value.toString()}
      </span>
    );
  }

  if (typeof value === 'number') {
    return <span className="text-blue-600">{value.toString()}</span>;
  }

  if (typeof value === 'string') {
    return <span className="text-gray-800">"{value}"</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-gray-500">[]</span>;
    }

    return (
      <div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-xs text-purple-600 hover:text-purple-800"
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span>Array ({value.length} items)</span>
        </button>
        {expanded && (
          <div className="ml-4 mt-1 space-y-1">
            {value.slice(0, 10).map((item, index) => (
              <div key={index} className="text-xs">
                <span className="text-gray-500">{index}:</span> <ValueDisplay value={item} />
              </div>
            ))}
            {value.length > 10 && (
              <div className="text-xs text-gray-500">
                ... and {value.length - 10} more
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  if (typeof value === 'object') {
    const keys = Object.keys(value);
    if (keys.length === 0) {
      return <span className="text-gray-500">{{}}</span>;
    }

    return (
      <div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-xs text-purple-600 hover:text-purple-800"
        >
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span>Object ({keys.length} keys)</span>
        </button>
        {expanded && (
          <div className="ml-4 mt-1 space-y-1">
            {keys.slice(0, 10).map((key) => (
              <div key={key} className="text-xs">
                <span className="text-gray-700 font-medium">{key}:</span>{' '}
                <ValueDisplay value={(value as Record<string, unknown>)[key]} />
              </div>
            ))}
            {keys.length > 10 && (
              <div className="text-xs text-gray-500">
                ... and {keys.length - 10} more keys
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  return <span className="text-gray-600">{String(value)}</span>;
}

/**
 * Shared context table component
 *
 * Displays the shared team context as a searchable table.
 * Supports nested objects and arrays with expand/collapse.
 */
export function SharedContextTable({
  context,
  className = '',
}: SharedContextTableProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const entries = Object.entries(context);

  // Filter entries by search term
  const filteredEntries = entries.filter(([key]) =>
    key.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className={`bg-white rounded-lg shadow border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Shared Context</h3>
        <p className="text-sm text-gray-600 mt-1">
          Team-wide key-value store ({entries.length} keys)
        </p>
      </div>

      {/* Search bar */}
      <div className="p-4 border-b border-gray-200">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search keys..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        {filteredEntries.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            {entries.length === 0 ? (
              <p className="text-sm">No context data available</p>
            ) : (
              <p className="text-sm">No keys match "{searchTerm}"</p>
            )}
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Key
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Type
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Value
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {filteredEntries.map(([key, value]) => (
                <tr key={key} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">{key}</td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {value === null
                      ? 'null'
                      : Array.isArray(value)
                      ? 'array'
                      : typeof value === 'object'
                      ? 'object'
                      : typeof value}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <ValueDisplay value={value} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 text-xs text-gray-500">
        Showing {filteredEntries.length} of {entries.length} keys
      </div>
    </div>
  );
}
