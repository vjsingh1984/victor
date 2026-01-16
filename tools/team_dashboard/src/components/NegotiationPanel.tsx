import React from 'react';
import { CheckCircle, XCircle, Users, MessageSquare } from 'lucide-react';
import type { NegotiationStatus } from '../types';

interface NegotiationPanelProps {
  status: NegotiationStatus | null;
  className?: string;
}

/**
 * Negotiation status badge
 */
function StatusBadge({ success, consensus }: { success: boolean; consensus: boolean }) {
  if (success && consensus) {
    return (
      <div className="flex items-center gap-1 px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
        <CheckCircle size={16} />
        <span>Consensus Reached</span>
      </div>
    );
  }

  if (success) {
    return (
      <div className="flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
        <CheckCircle size={16} />
        <span>Agreement Reached</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1 px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium">
      <XCircle size={16} />
      <span>No Agreement</span>
    </div>
  );
}

/**
 * Vote visualization bar
 */
function VoteBar({ votes }: { votes: Record<string, unknown> }) {
  const voteEntries = Object.entries(votes);

  if (voteEntries.length === 0) {
    return <p className="text-sm text-gray-500">No votes recorded</p>;
  }

  const maxValue = Math.max(
    ...voteEntries.map(([, value]) => (typeof value === 'number' ? value : 0))
  );

  return (
    <div className="space-y-2">
      {voteEntries.map(([proposal, value]) => {
        const count = typeof value === 'number' ? value : 0;
        const percentage = maxValue > 0 ? (count / maxValue) * 100 : 0;

        return (
          <div key={proposal}>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-gray-700">{proposal}</span>
              <span className="text-gray-500">{count}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Negotiation panel component
 *
 * Displays negotiation status including:
 * - Success/failure and consensus status
 * - Number of rounds
 * - Vote distribution
 * - Agreed proposal (if any)
 */
export function NegotiationPanel({ status, className = '' }: NegotiationPanelProps) {
  if (!status) {
    return (
      <div
        className={`bg-white rounded-lg shadow p-6 border border-gray-200 ${className}`}
      >
        <div className="text-center text-gray-500">
          <MessageSquare className="w-12 h-12 mx-auto mb-2 text-gray-300" />
          <p className="text-sm">No negotiation data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Negotiation Status</h3>
          <StatusBadge success={status.success} consensus={status.consensus_achieved} />
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Rounds */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <MessageSquare className="w-4 h-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-700">Rounds</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">{status.rounds}</p>
        </div>

        {/* Consensus indicator */}
        {status.consensus_achieved !== null && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-2">
              <Users className="w-5 h-5 text-blue-600" />
              <div>
                <p className="text-sm font-medium text-blue-900">Full Consensus</p>
                <p className="text-xs text-blue-700">
                  {status.consensus_achieved
                    ? 'All members agreed on the proposal'
                    : 'Decision reached without full consensus'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Agreed proposal */}
        {status.agreed_proposal && (
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Agreed Proposal</p>
            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <p className="text-sm text-gray-900">{status.agreed_proposal.content}</p>
              <p className="text-xs text-gray-500 mt-2">
                Proposed by: {status.agreed_proposal.proposer_id}
              </p>
            </div>
          </div>
        )}

        {/* Votes */}
        {Object.keys(status.votes).length > 0 && (
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Vote Distribution</p>
            <VoteBar votes={status.votes} />
          </div>
        )}
      </div>
    </div>
  );
}
