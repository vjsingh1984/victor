/**
 * Node grouping component
 */

import React, { useState } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { Group, Ungroup, Minus2, Plus } from 'lucide-react';

export const NodeGroupingControls: React.FC = () => {
  const {
    selectedNodes,
    nodes,
    nodeGroups,
    createGroup,
    deleteGroup,
    toggleGroupCollapse,
  } = useWorkflowStore();

  const [showGroupDialog, setShowGroupDialog] = useState(false);
  const [groupLabel, setGroupLabel] = useState('');
  const [groupColor, setGroupColor] = useState('#8b5cf6');

  const handleCreateGroup = () => {
    if (selectedNodes.length < 2) {
      alert('Please select at least 2 nodes to create a group');
      return;
    }

    setShowGroupDialog(true);
  };

  const handleConfirmGroup = () => {
    if (!groupLabel.trim()) {
      alert('Please enter a group label');
      return;
    }

    createGroup(selectedNodes, groupLabel, groupColor);
    setShowGroupDialog(false);
    setGroupLabel('');
    setGroupColor('#8b5cf6');
  };

  const colors = [
    '#8b5cf6', // purple
    '#3b82f6', // blue
    '#10b981', // green
    '#f59e0b', // amber
    '#ef4444', // red
    '#ec4899', // pink
    '#6366f1', // indigo
    '#14b8a6', // teal
  ];

  return (
    <>
      <div className="absolute top-4 right-4 bg-white rounded-lg shadow-lg border border-slate-200 p-2 z-10">
        <div className="flex items-center gap-2">
          {selectedNodes.length >= 2 && (
            <button
              onClick={handleCreateGroup}
              className="flex items-center gap-2 px-3 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg transition-colors"
              title="Group selected nodes"
            >
              <Group className="w-4 h-4" />
              <span className="text-sm">Group</span>
            </button>
          )}

          {Object.keys(nodeGroups).length > 0 && (
            <div className="border-l border-slate-200 pl-2">
              <span className="text-xs text-slate-600">
                {Object.keys(nodeGroups).length} group
                {Object.keys(nodeGroups).length !== 1 ? 's' : ''}
              </span>
            </div>
          )}
        </div>
      </div>

      {showGroupDialog && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setShowGroupDialog(false)}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-md w-full p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold text-slate-800 mb-4">
              Create Group
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Group Label
                </label>
                <input
                  type="text"
                  value={groupLabel}
                  onChange={(e) => setGroupLabel(e.target.value)}
                  placeholder="e.g., Authentication Flow"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Group Color
                </label>
                <div className="flex flex-wrap gap-2">
                  {colors.map((color) => (
                    <button
                      key={color}
                      onClick={() => setGroupColor(color)}
                      className={`w-8 h-8 rounded-full border-2 transition-all ${
                        groupColor === color
                          ? 'border-purple-600 scale-110'
                          : 'border-transparent hover:scale-105'
                      }`}
                      style={{ backgroundColor: color }}
                    />
                  ))}
                </div>
              </div>

              <div className="pt-4 flex gap-2">
                <button
                  onClick={handleConfirmGroup}
                  className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                >
                  Create Group
                </button>
                <button
                  onClick={() => setShowGroupDialog(false)}
                  className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// Group badge component to show on nodes
export const GroupBadge: React.FC<{
  groupId: string;
  label: string;
  color: string;
  onCollapse?: () => void;
  onExpand?: () => void;
  collapsed?: boolean;
}> = ({ groupId, label, color, onCollapse, onExpand, collapsed }) => {
  return (
    <div
      className="absolute -top-3 -left-3 px-2 py-1 rounded-full text-xs font-medium text-white shadow-lg cursor-pointer"
      style={{ backgroundColor: color }}
      onClick={collapsed ? onExpand : onCollapse}
    >
      <div className="flex items-center gap-1">
        {collapsed ? <Plus className="w-3 h-3" /> : <Minus2 className="w-3 h-3" />}
        <span>{label}</span>
      </div>
    </div>
  );
};
