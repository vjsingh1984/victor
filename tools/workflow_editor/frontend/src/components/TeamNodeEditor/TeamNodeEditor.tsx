/**
 * Team Node Editor - Configure team members and formation
 */

import React, { useState, useEffect } from 'react';
import { X, Plus, Trash2, Save } from 'lucide-react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { TeamMember, TeamFormation } from '../../types';
import FormationSelector from '../FormationSelector/FormationSelector';
import MemberForm from './MemberForm';
import './TeamNodeEditor.css';

interface TeamNodeEditorProps {
  onClose: () => void;
}

const TeamNodeEditor: React.FC<TeamNodeEditorProps> = ({ onClose }) => {
  const { selectedNode, updateNode } = useWorkflowStore();
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [formation, setFormation] = useState<TeamFormation>('parallel');
  const [goal, setGoal] = useState('');
  const [maxIterations, setMaxIterations] = useState(5);
  const [timeout, setTimeout] = useState<number | undefined>(undefined);
  const [editingMember, setEditingMember] = useState<TeamMember | null>(null);

  // Load existing team configuration
  useEffect(() => {
    if (selectedNode?.data) {
      setMembers(selectedNode.data.members || []);
      setFormation(selectedNode.data.formation || 'parallel');
      setGoal(selectedNode.data.goal || '');
      setMaxIterations(selectedNode.data.max_iterations || 5);
      setTimeout(selectedNode.data.timeout_seconds);
    }
  }, [selectedNode]);

  const handleSave = () => {
    if (selectedNode) {
      updateNode(selectedNode.id, {
        members,
        formation,
        goal,
        max_iterations: maxIterations,
        timeout_seconds: timeout,
      });
    }
    onClose();
  };

  const handleAddMember = () => {
    const newMember: TeamMember = {
      id: `member_${Date.now()}`,
      role: 'assistant',
      goal: '',
      tool_budget: 25,
    };
    setMembers([...members, newMember]);
    setEditingMember(newMember);
  };

  const handleEditMember = (member: TeamMember) => {
    setEditingMember(member);
  };

  const handleDeleteMember = (memberId: string) => {
    setMembers(members.filter((m) => m.id !== memberId));
    if (editingMember?.id === memberId) {
      setEditingMember(null);
    }
  };

  const handleUpdateMember = (updatedMember: TeamMember) => {
    setMembers(
      members.map((m) => (m.id === updatedMember.id ? updatedMember : m))
    );
    setEditingMember(null);
  };

  if (!selectedNode) return null;

  return (
    <div className="team-node-editor-overlay" onClick={onClose}>
      <div
        className="team-node-editor"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="team-node-editor-header">
          <h2 className="text-xl font-bold text-slate-800">
            Edit Team Node
          </h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-100 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="team-node-editor-content">
          {/* Formation Selection */}
          <div className="form-section">
            <label className="form-label">Team Formation</label>
            <FormationSelector
              selected={formation}
              onSelect={setFormation}
            />
          </div>

          {/* Team Goal */}
          <div className="form-section">
            <label className="form-label">Team Goal</label>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="What should this team accomplish?"
              className="form-textarea"
              rows={3}
            />
          </div>

          {/* Iteration Settings */}
          <div className="form-row">
            <div className="form-field">
              <label className="form-label">Max Iterations</label>
              <input
                type="number"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                className="form-input"
                min={1}
                max={25}
              />
            </div>
            <div className="form-field">
              <label className="form-label">Timeout (seconds)</label>
              <input
                type="number"
                value={timeout || ''}
                onChange={(e) =>
                  setTimeout(
                    e.target.value ? parseInt(e.target.value) : undefined
                  )
                }
                className="form-input"
                min={1}
                placeholder="Optional"
              />
            </div>
          </div>

          {/* Team Members */}
          <div className="form-section">
            <div className="flex items-center justify-between mb-3">
              <label className="form-label">Team Members</label>
              <button
                onClick={handleAddMember}
                className="flex items-center gap-1 px-3 py-1.5 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm"
              >
                <Plus className="w-4 h-4" />
                Add Member
              </button>
            </div>

            {/* Member List */}
            <div className="member-list">
              {members.length === 0 ? (
                <div className="member-list-empty">
                  <p className="text-slate-500 text-sm">
                    No team members yet. Add members to get started.
                  </p>
                </div>
              ) : (
                members.map((member) => (
                  <div
                    key={member.id}
                    className="member-list-item team-member-item"
                  >
                    <div className="member-info">
                      <div className="flex items-center gap-2">
                        <span className="member-avatar">
                          {member.role[0]?.toUpperCase()}
                        </span>
                        <div>
                          <div className="font-medium text-sm">
                            {member.role}
                          </div>
                          <div className="text-xs text-slate-600">
                            {member.goal || 'No goal set'}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="member-actions">
                      <button
                        onClick={() => handleEditMember(member)}
                        className="p-1.5 hover:bg-slate-100 rounded"
                        title="Edit member"
                      >
                        <Plus className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDeleteMember(member.id)}
                        className="p-1.5 hover:bg-red-100 text-red-600 rounded"
                        title="Delete member"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="team-node-editor-footer">
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            <Save className="w-4 h-4" />
            Save Changes
          </button>
        </div>

        {/* Member Form Modal */}
        {editingMember && (
          <MemberForm
            member={editingMember}
            onSave={handleUpdateMember}
            onCancel={() => setEditingMember(null)}
          />
        )}
      </div>
    </div>
  );
};

export default TeamNodeEditor;
