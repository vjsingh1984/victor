/**
 * YAML Preview Panel - Real-time YAML preview
 */

import React, { useEffect, useState } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { X, Copy, Download } from 'lucide-react';
import YAML from 'yaml';
import './YAMLPreview.css';

interface YAMLPreviewProps {
  onClose: () => void;
}

const YAMLPreview: React.FC<YAMLPreviewProps> = ({ onClose }) => {
  const { nodes, edges } = useWorkflowStore();
  const [yamlContent, setYamlContent] = useState('');

  useEffect(() => {
    // Convert workflow to YAML format
    const workflow = {
      workflows: {
        example: {
          description: 'Workflow created with Victor Editor',
          metadata: {
            version: '1.0',
            created_at: new Date().toISOString(),
          },
          nodes: nodes.map((node) => ({
            id: node.id,
            type: node.type,
            name: node.data.label || node.type,
            ...node.data,
          })),
          edges: edges.map((edge) => ({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label,
          })),
        },
      },
    };

    try {
      const yaml = YAML.stringify(workflow);
      setYamlContent(yaml);
    } catch (error) {
      console.error('Failed to convert to YAML:', error);
    }
  }, [nodes, edges]);

  const handleCopy = () => {
    navigator.clipboard.writeText(yamlContent);
    alert('YAML copied to clipboard!');
  };

  const handleDownload = () => {
    const blob = new Blob([yamlContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'workflow.yaml';
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="yaml-preview-overlay" onClick={onClose}>
      <div
        className="yaml-preview-panel"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="yaml-preview-header">
          <h2 className="yaml-preview-title">YAML Preview</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="yaml-preview-button"
              title="Copy to clipboard"
            >
              <Copy className="w-4 h-4" />
              Copy
            </button>
            <button
              onClick={handleDownload}
              className="yaml-preview-button"
              title="Download YAML"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
            <button
              onClick={onClose}
              className="yaml-preview-button"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="yaml-preview-content">
          <pre className="yaml-preview-code">{yamlContent}</pre>
        </div>
      </div>
    </div>
  );
};

export default YAMLPreview;
