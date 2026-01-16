/**
 * Team Member Form - Add/Edit team member
 */

import React, { useState, useEffect } from 'react';
import { X, Save } from 'lucide-react';
import { TeamMember } from '../../types';
import './TeamNodeEditor.css';

interface MemberFormProps {
  member: TeamMember;
  onSave: (member: TeamMember) => void;
  onCancel: () => void;
}

const ROLE_OPTIONS = [
  { value: 'assistant', label: 'Assistant' },
  { value: 'researcher', label: 'Researcher' },
  { value: 'planner', label: 'Planner' },
  { value: 'executor', label: 'Executor' },
  { value: 'reviewer', label: 'Reviewer' },
  { value: 'writer', label: 'Writer' },
];

const EXPERTISE_OPTIONS = [
  'python',
  'javascript',
  'typescript',
  'security',
  'performance',
  'testing',
  'documentation',
  'devops',
  'data-analysis',
  'ml',
];

const MemberForm: React.FC<MemberFormProps> = ({ member, onSave, onCancel }) => {
  const [formData, setFormData] = useState<TeamMember>(member);

  useEffect(() => {
    setFormData(member);
  }, [member]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  const toggleExpertise = (expertise: string) => {
    const current = formData.expertise || [];
    if (current.includes(expertise)) {
      setFormData({
        ...formData,
        expertise: current.filter((e) => e !== expertise),
      });
    } else {
      setFormData({
        ...formData,
        expertise: [...current, expertise],
      });
    }
  };

  return (
    <div className="team-node-editor-overlay" onClick={onCancel}>
      <div
        className="team-node-editor"
        style={{ maxWidth: '600px' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="team-node-editor-header">
          <h2 className="text-xl font-bold text-slate-800">
            {member.goal ? 'Edit Member' : 'Add Member'}
          </h2>
          <button
            onClick={onCancel}
            className="p-1 hover:bg-slate-100 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="team-node-editor-content">
          {/* Role */}
          <div className="form-section">
            <label className="form-label">Role</label>
            <select
              value={formData.role}
              onChange={(e) =>
                setFormData({ ...formData, role: e.target.value })
              }
              className="form-input"
              required
            >
              {ROLE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Goal */}
          <div className="form-section">
            <label className="form-label">Goal</label>
            <textarea
              value={formData.goal}
              onChange={(e) =>
                setFormData({ ...formData, goal: e.target.value })
              }
              placeholder="What is this member responsible for?"
              className="form-textarea"
              rows={3}
              required
            />
          </div>

          {/* Tool Budget */}
          <div className="form-section">
            <label className="form-label">Tool Budget</label>
            <input
              type="number"
              value={formData.tool_budget}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  tool_budget: parseInt(e.target.value),
                })
              }
              className="form-input"
              min={1}
              max={100}
              required
            />
          </div>

          {/* Backstory */}
          <div className="form-section">
            <label className="form-label">Backstory (Optional)</label>
            <textarea
              value={formData.backstory || ''}
              onChange={(e) =>
                setFormData({ ...formData, backstory: e.target.value })
              }
              placeholder="Background context for this member"
              className="form-textarea"
              rows={2}
            />
          </div>

          {/* Expertise Tags */}
          <div className="form-section">
            <label className="form-label">Areas of Expertise</label>
            <div className="flex flex-wrap gap-2 mt-2">
              {EXPERTISE_OPTIONS.map((expertise) => (
                <button
                  key={expertise}
                  type="button"
                  onClick={() => toggleExpertise(expertise)}
                  className={`
                    px-3 py-1 rounded-full text-sm font-medium transition-colors
                    ${
                      (formData.expertise || []).includes(expertise)
                        ? 'bg-purple-600 text-white'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                    }
                  `}
                >
                  {expertise}
                </button>
              ))}
            </div>
          </div>

          {/* Personality */}
          <div className="form-section">
            <label className="form-label">Personality (Optional)</label>
            <input
              type="text"
              value={formData.personality || ''}
              onChange={(e) =>
                setFormData({ ...formData, personality: e.target.value })
              }
              placeholder="e.g., thorough and detail-oriented"
              className="form-input"
            />
          </div>

          {/* Footer */}
          <div className="team-node-editor-footer">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
            >
              <Save className="w-4 h-4" />
              Save Member
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default MemberForm;
