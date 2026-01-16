/**
 * Template library sidebar
 */

import React, { useState, useEffect } from 'react';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import { X, ChevronDown, ChevronRight, Folder } from 'lucide-react';
import templatesData from '../../templates/templates.json';

interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  nodes: any[];
  edges: any[];
}

export const TemplateLibrary: React.FC<{
  onClose: () => void;
  onInsertTemplate: (template: Template) => void;
}> = ({ onClose, onInsertTemplate }) => {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Record<string, boolean>>(
    {}
  );

  // Group templates by category
  const categories: Record<string, Template[]> = {};
  templatesData.templates.forEach((template: Template) => {
    if (!categories[template.category]) {
      categories[template.category] = [];
    }
    categories[template.category].push(template);
  });

  // Auto-expand first category
  useEffect(() => {
    const categoryKeys = Object.keys(categories);
    if (categoryKeys.length > 0 && !selectedCategory) {
      setSelectedCategory(categoryKeys[0]);
      setExpandedCategories({ [categoryKeys[0]]: true });
    }
  }, [categories, selectedCategory]);

  const toggleCategory = (category: string) => {
    setExpandedCategories({
      ...expandedCategories,
      [category]: !expandedCategories[category],
    });
    setSelectedCategory(category);
  };

  const handleInsertTemplate = (template: Template) => {
    onInsertTemplate(template);
    onClose();
  };

  return (
    <div className="fixed left-0 top-0 h-full w-80 bg-white border-r border-slate-200 shadow-lg z-40 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-800">Template Library</h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-slate-100 rounded transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Categories */}
      <div className="p-4">
        <div className="text-sm text-slate-600 mb-2">Categories</div>
        <div className="space-y-1">
          {Object.keys(categories).map((category) => (
            <div key={category}>
              <button
                onClick={() => toggleCategory(category)}
                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-slate-100 rounded-lg transition-colors text-left"
              >
                {expandedCategories[category] ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
                <Folder className="w-4 h-4" />
                <span className="capitalize font-medium">{category}</span>
                <span className="ml-auto text-xs text-slate-500">
                  {categories[category].length}
                </span>
              </button>

              {expandedCategories[category] && (
                <div className="ml-6 mt-1 space-y-2">
                  {categories[category].map((template) => (
                    <TemplateCard
                      key={template.id}
                      template={template}
                      onInsert={() => handleInsertTemplate(template)}
                    />
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const TemplateCard: React.FC<{
  template: Template;
  onInsert: () => void;
}> = ({ template, onInsert }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="border border-slate-200 rounded-lg p-3 hover:border-purple-300 transition-colors">
      <div className="flex items-start gap-2">
        <span className="text-2xl">{template.icon}</span>
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-slate-800 text-sm">{template.name}</h4>
          <p className="text-xs text-slate-600 mt-1 line-clamp-2">
            {template.description}
          </p>
          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={onInsert}
              className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded hover:bg-purple-200 transition-colors"
            >
              Insert
            </button>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-xs px-2 py-1 text-slate-600 hover:bg-slate-100 rounded transition-colors"
            >
              {isExpanded ? 'Hide' : 'Details'}
            </button>
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-slate-200">
          <div className="text-xs text-slate-600 mb-2">
            <strong>Nodes:</strong> {template.nodes.length}
          </div>
          <div className="space-y-1">
            {template.nodes.slice(0, 5).map((node, index) => (
              <div key={index} className="text-xs text-slate-600 flex items-center gap-2">
                <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                <span className="truncate">{node.data.label}</span>
                <span className="text-slate-400">({node.type})</span>
              </div>
            ))}
            {template.nodes.length > 5 && (
              <div className="text-xs text-slate-500">
                +{template.nodes.length - 5} more nodes
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Template button for header
export const TemplateLibraryButton: React.FC<{
  onClick: () => void;
}> = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-lg transition-colors"
      title="Open template library"
    >
      <Folder className="w-4 h-4" />
      Templates
    </button>
  );
};
