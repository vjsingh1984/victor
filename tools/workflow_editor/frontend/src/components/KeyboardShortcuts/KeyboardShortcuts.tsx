/**
 * Keyboard shortcuts handler
 */

import React, { useEffect, useCallback } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';

interface KeyboardShortcutsProps {
  onHelp?: () => void;
}

export const KeyboardShortcuts: React.FC<KeyboardShortcutsProps> = ({ onHelp }) => {
  const {
    undo,
    redo,
    canUndo,
    canRedo,
    selectedNode,
    deleteNode,
    duplicateNode,
    copyNodes,
    pasteNodes,
    selectedNodes,
    selectNodes,
    nodes,
    fitToScreen,
    zoomIn,
    zoomOut,
    resetZoom,
    searchQuery,
    setSearchQuery,
    nextMatch,
    previousMatch,
    clearSearch,
  } = useWorkflowStore();

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Ignore if typing in input field
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.contentEditable === 'true'
      ) {
        return;
      }

      // Ctrl+Z: Undo
      if (event.ctrlKey && event.key === 'z' && !event.shiftKey) {
        event.preventDefault();
        if (canUndo()) {
          undo();
        }
        return;
      }

      // Ctrl+Y or Ctrl+Shift+Z: Redo
      if (
        (event.ctrlKey && event.key === 'y') ||
        (event.ctrlKey && event.shiftKey && event.key === 'z')
      ) {
        event.preventDefault();
        if (canRedo()) {
          redo();
        }
        return;
      }

      // Ctrl+S: Save/Export
      if (event.ctrlKey && event.key === 's') {
        event.preventDefault();
        // Trigger export
        const exportEvent = new CustomEvent('workflow-export');
        window.dispatchEvent(exportEvent);
        return;
      }

      // Ctrl+C: Copy
      if (event.ctrlKey && event.key === 'c') {
        if (selectedNode || selectedNodes.length > 0) {
          event.preventDefault();
          const ids = selectedNodes.length > 0 ? selectedNodes : [selectedNode?.id || ''];
          copyNodes(ids);
        }
        return;
      }

      // Ctrl+V: Paste
      if (event.ctrlKey && event.key === 'v') {
        event.preventDefault();
        pasteNodes();
        return;
      }

      // Ctrl+D: Duplicate
      if (event.ctrlKey && event.key === 'd') {
        if (selectedNode) {
          event.preventDefault();
          duplicateNode(selectedNode.id);
        }
        return;
      }

      // Ctrl+A: Select all
      if (event.ctrlKey && event.key === 'a') {
        event.preventDefault();
        const allIds = nodes.map((n) => n.id);
        selectNodes(allIds);
        return;
      }

      // Delete/Backspace: Delete selected
      if (event.key === 'Delete' || event.key === 'Backspace') {
        if (selectedNode || selectedNodes.length > 0) {
          event.preventDefault();
          const ids = selectedNodes.length > 0 ? selectedNodes : [selectedNode?.id || ''];
          ids.forEach((id) => {
            if (id) deleteNode(id);
          });
        }
        return;
      }

      // Escape: Deselect
      if (event.key === 'Escape') {
        event.preventDefault();
        selectNodes([]);
        clearSearch();
        return;
      }

      // Ctrl+F: Focus search
      if (event.ctrlKey && event.key === 'f') {
        event.preventDefault();
        const searchInput = document.querySelector(
          'input[placeholder*="Search nodes"]'
        ) as HTMLInputElement;
        searchInput?.focus();
        return;
      }

      // F3 or Ctrl+G: Next search result
      if (event.key === 'F3' || (event.ctrlKey && event.key === 'g')) {
        event.preventDefault();
        if (searchQuery) {
          nextMatch();
        }
        return;
      }

      // Shift+F3 or Ctrl+Shift+G: Previous search result
      if (
        (event.key === 'F3' && event.shiftKey) ||
        (event.ctrlKey && event.shiftKey && event.key === 'g')
      ) {
        event.preventDefault();
        if (searchQuery) {
          previousMatch();
        }
        return;
      }

      // Ctrl+0: Reset zoom
      if (event.ctrlKey && event.key === '0') {
        event.preventDefault();
        resetZoom();
        return;
      }

      // Ctrl++ or Ctrl+=: Zoom in
      if (event.ctrlKey && (event.key === '+' || event.key === '=')) {
        event.preventDefault();
        zoomIn();
        return;
      }

      // Ctrl+-: Zoom out
      if (event.ctrlKey && event.key === '-') {
        event.preventDefault();
        zoomOut();
        return;
      }

      // Ctrl+9: Fit to screen
      if (event.ctrlKey && event.key === '9') {
        event.preventDefault();
        fitToScreen();
        return;
      }

      // Ctrl+/: Show keyboard shortcuts help
      if (event.ctrlKey && event.key === '/') {
        event.preventDefault();
        onHelp?.();
        return;
      }
    },
    [
      undo,
      redo,
      canUndo,
      canRedo,
      selectedNode,
      deleteNode,
      duplicateNode,
      copyNodes,
      pasteNodes,
      selectedNodes,
      selectNodes,
      nodes,
      fitToScreen,
      zoomIn,
      zoomOut,
      resetZoom,
      searchQuery,
      setSearchQuery,
      nextMatch,
      previousMatch,
      clearSearch,
      onHelp,
    ]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  return null;
};

// Keyboard shortcuts help modal
export const KeyboardShortcutsHelp: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-auto p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-slate-800">Keyboard Shortcuts</h2>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 text-2xl"
          >
            ×
          </button>
        </div>

        <div className="space-y-6">
          <section>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Edit Operations</h3>
            <div className="grid grid-cols-2 gap-2">
              <ShortcutRow keys={['Ctrl', 'Z']} description="Undo last operation" />
              <ShortcutRow keys={['Ctrl', 'Y']} description="Redo last operation" />
              <ShortcutRow keys={['Ctrl', 'C']} description="Copy selected nodes" />
              <ShortcutRow keys={['Ctrl', 'V']} description="Paste nodes" />
              <ShortcutRow keys={['Ctrl', 'D']} description="Duplicate selected node" />
              <ShortcutRow keys={['Delete']} description="Delete selected nodes" />
            </div>
          </section>

          <section>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Selection</h3>
            <div className="grid grid-cols-2 gap-2">
              <ShortcutRow keys={['Ctrl', 'A']} description="Select all nodes" />
              <ShortcutRow keys={['Escape']} description="Deselect all / Close panels" />
            </div>
          </section>

          <section>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Zoom & Pan</h3>
            <div className="grid grid-cols-2 gap-2">
              <ShortcutRow keys={['Ctrl', '0']} description="Reset zoom to 100%" />
              <ShortcutRow keys={['Ctrl', '+']} description="Zoom in" />
              <ShortcutRow keys={['Ctrl', '-']} description="Zoom out" />
              <ShortcutRow keys={['Ctrl', '9']} description="Fit to screen" />
              <ShortcutRow keys={['Mouse Wheel']} description="Zoom in/out" />
              <ShortcutRow keys={['Click + Drag']} description="Pan canvas" />
            </div>
          </section>

          <section>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Search</h3>
            <div className="grid grid-cols-2 gap-2">
              <ShortcutRow keys={['Ctrl', 'F']} description="Focus search bar" />
              <ShortcutRow keys={['F3', 'Ctrl + G']} description="Next search result" />
              <ShortcutRow keys={['Shift', 'F3']} description="Previous search result" />
            </div>
          </section>

          <section>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">File Operations</h3>
            <div className="grid grid-cols-2 gap-2">
              <ShortcutRow keys={['Ctrl', 'S']} description="Save/Export workflow" />
              <ShortcutRow keys={['Ctrl', 'O']} description="Import workflow" />
            </div>
          </section>

          <section>
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Help</h3>
            <div className="grid grid-cols-2 gap-2">
              <ShortcutRow keys={['Ctrl', '/']} description="Show this help" />
            </div>
          </section>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-200">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            Got it!
          </button>
        </div>
      </div>
    </div>
  );
};

const ShortcutRow: React.FC<{ keys: string[]; description: string }> = ({
  keys,
  description,
}) => {
  return (
    <div className="flex items-center gap-2 text-sm">
      <div className="flex gap-1">
        {keys.map((key, index) => (
          <React.Fragment key={index}>
            <kbd className="px-2 py-1 bg-slate-100 border border-slate-300 rounded text-xs font-mono">
              {key}
            </kbd>
            {index < keys.length - 1 && <span className="text-slate-400">+</span>}
          </React.Fragment>
        ))}
      </div>
      <span className="text-slate-600 ml-2">{description}</span>
    </div>
  );
};
