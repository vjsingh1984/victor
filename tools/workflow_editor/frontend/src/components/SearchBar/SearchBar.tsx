/**
 * Search bar for finding nodes
 */

import React, { useState, useEffect } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { Search, ChevronLeft, ChevronRight, X } from 'lucide-react';

export const SearchBar: React.FC = () => {
  const {
    searchQuery,
    setSearchQuery,
    searchMatches,
    currentMatchIndex,
    nextMatch,
    previousMatch,
    clearSearch,
    selectNode,
    nodes,
  } = useWorkflowStore();

  const [localQuery, setLocalQuery] = useState(searchQuery);

  useEffect(() => {
    setLocalQuery(searchQuery);
  }, [searchQuery]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchQuery(localQuery);
  };

  const handleNext = () => {
    if (searchMatches.length === 0) return;

    const nextIndex = (currentMatchIndex + 1) % searchMatches.length;
    const nodeId = searchMatches[nextIndex];
    const node = nodes.find((n) => n.id === nodeId);
    if (node) {
      selectNode(node);
    }
    nextMatch();
  };

  const handlePrevious = () => {
    if (searchMatches.length === 0) return;

    const prevIndex =
      (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length;
    const nodeId = searchMatches[prevIndex];
    const node = nodes.find((n) => n.id === nodeId);
    if (node) {
      selectNode(node);
    }
    previousMatch();
  };

  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-white border-b border-slate-200">
      <form onSubmit={handleSubmit} className="flex items-center gap-2 flex-1">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search nodes by ID, name, or type..."
            className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            value={localQuery}
            onChange={(e) => setLocalQuery(e.target.value)}
          />
          {localQuery && (
            <button
              type="button"
              onClick={() => {
                setLocalQuery('');
                clearSearch();
              }}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </form>

      {searchMatches.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-600">
            {currentMatchIndex + 1} of {searchMatches.length}
          </span>
          <button
            onClick={handlePrevious}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="Previous match (Shift+F3)"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={handleNext}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="Next match (F3)"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}

      {searchQuery && searchMatches.length === 0 && (
        <span className="text-sm text-slate-500">No matches found</span>
      )}
    </div>
  );
};
