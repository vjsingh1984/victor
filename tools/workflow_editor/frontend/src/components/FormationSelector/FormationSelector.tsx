/**
 * Formation Selector - Visual team formation selection
 */

import React, { useState, useEffect } from 'react';
import { TeamFormation, FormationInfo } from '../../types';
import './FormationSelector.css';

interface FormationSelectorProps {
  selected: TeamFormation;
  onSelect: (formation: TeamFormation) => void;
}

const FORMATIONS: Record<TeamFormation, FormationInfo> = {
  parallel: {
    name: 'Parallel',
    description: 'All members work simultaneously on the task',
    icon: '||',
    best_for: ['Independent analysis', 'Multi-perspective review'],
    communication_style: 'structured',
  },
  sequential: {
    name: 'Sequential',
    description: 'Members work in sequence, each building on the previous',
    icon: '→',
    best_for: ['Step-by-step processing', 'Refinement loops'],
    communication_style: 'pass_through',
  },
  pipeline: {
    name: 'Pipeline',
    description: 'Output passes through stages like an assembly line',
    icon: '⇒',
    best_for: ['Multi-stage processing', 'Specialized tasks'],
    communication_style: 'structured',
  },
  hierarchical: {
    name: 'Hierarchical',
    description: 'Manager coordinates worker agents',
    icon: '⬗',
    best_for: ['Complex coordination', 'Task delegation'],
    communication_style: 'coordinated',
  },
  consensus: {
    name: 'Consensus',
    description: 'Members vote on decisions',
    icon: '◊',
    best_for: ['Decision making', 'Quality assurance'],
    communication_style: 'peer_to_peer',
  },
};

const FormationSelector: React.FC<FormationSelectorProps> = ({
  selected,
  onSelect,
}) => {
  const [hovered, setHovered] = useState<TeamFormation | null>(null);

  return (
    <div className="formation-selector">
      <div className="formation-grid">
        {Object.entries(FORMATIONS).map(([key, info]) => (
          <button
            key={key}
            className={`
              formation-card
              ${selected === key ? 'formation-card-selected' : ''}
              ${hovered === key ? 'formation-card-hovered' : ''}
            `}
            onClick={() => onSelect(key as TeamFormation)}
            onMouseEnter={() => setHovered(key as TeamFormation)}
            onMouseLeave={() => setHovered(null)}
          >
            {/* Icon */}
            <div className="formation-icon">{info.icon}</div>

            {/* Name */}
            <div className="formation-name">{info.name}</div>

            {/* Description */}
            <div className="formation-description">{info.description}</div>

            {/* Best for badges */}
            <div className="formation-best-for">
              {info.best_for.map((use) => (
                <span key={use} className="formation-badge">
                  {use}
                </span>
              ))}
            </div>

            {/* Visual diagram */}
            {hovered === key && (
              <div className="formation-diagram">
                <FormationDiagram type={key as TeamFormation} />
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
};

// Formation diagram component
const FormationDiagram: React.FC<{ type: TeamFormation }> = ({ type }) => {
  switch (type) {
    case 'parallel':
      return (
        <div className="diagram-parallel">
          <div className="diagram-input">Input</div>
          <div className="diagram-parallel-branches">
            <div className="diagram-branch">Member 1</div>
            <div className="diagram-branch">Member 2</div>
            <div className="diagram-branch">Member 3</div>
          </div>
          <div className="diagram-output">Output</div>
        </div>
      );

    case 'sequential':
      return (
        <div className="diagram-sequential">
          <div className="diagram-input">Input</div>
          <div className="diagram-flow">
            <div>Member 1</div>
            <div className="diagram-arrow">→</div>
            <div>Member 2</div>
            <div className="diagram-arrow">→</div>
            <div>Member 3</div>
          </div>
          <div className="diagram-output">Output</div>
        </div>
      );

    case 'pipeline':
      return (
        <div className="diagram-pipeline">
          <div className="diagram-input">Input</div>
          <div className="diagram-pipeline-stages">
            <div className="diagram-stage">Stage 1</div>
            <div className="diagram-arrow">→</div>
            <div className="diagram-stage">Stage 2</div>
            <div className="diagram-arrow">→</div>
            <div className="diagram-stage">Stage 3</div>
          </div>
          <div className="diagram-output">Output</div>
        </div>
      );

    case 'hierarchical':
      return (
        <div className="diagram-hierarchical">
          <div className="diagram-input">Input</div>
          <div className="diagram-hierarchy">
            <div className="diagram-manager">Manager</div>
            <div className="diagram-workers">
              <div className="diagram-worker">Worker 1</div>
              <div className="diagram-worker">Worker 2</div>
              <div className="diagram-worker">Worker 3</div>
            </div>
          </div>
          <div className="diagram-output">Output</div>
        </div>
      );

    case 'consensus':
      return (
        <div className="diagram-consensus">
          <div className="diagram-input">Input</div>
          <div className="diagram-consensus-members">
            <div className="diagram-member">Member 1</div>
            <div className="diagram-member">Member 2</div>
            <div className="diagram-member">Member 3</div>
          </div>
          <div className="diagram-vote">Vote →</div>
          <div className="diagram-output">Output</div>
        </div>
      );

    default:
      return null;
  }
};

export default FormationSelector;
