"""
Victor Coding Tools

25 coding-specific tools for code analysis, search, and modification.
These tools are registered via entry points in pyproject.toml.

Usage:
    from victor_coding.tools import CodeSearchTool, CodeReviewTool

    # Or access via entry point discovery
    from victor.verticals.vertical_loader import discover_tool_plugins
    tools = discover_tool_plugins()
"""

__version__ = "0.1.0"

# Coding tool mappings to victor.tools module
# These will be re-exported from the main package during migration
_CODING_TOOL_MAPPINGS = {
    # Code Search & Analysis
    "CodeSearchTool": "victor.tools.code_search_tool",
    "CodeIntelligenceTool": "victor.tools.code_intelligence_tool",
    "LanguageAnalyzerTool": "victor.tools.language_analyzer",
    "GraphTool": "victor.tools.graph_tool",
    "ArchitectureSummaryTool": "victor.tools.architecture_summary",

    # Code Review & Quality
    "CodeReviewTool": "victor.tools.code_review_tool",
    "AuditTool": "victor.tools.audit_tool",
    "MetricsTool": "victor.tools.metrics_tool",

    # Code Modification
    "FileEditorTool": "victor.tools.file_editor_tool",
    "PatchTool": "victor.tools.patch_tool",
    "MergeTool": "victor.tools.merge_tool",

    # LSP & Language Server
    "LSPTool": "victor.tools.lsp_tool",

    # Documentation
    "DocumentationTool": "victor.tools.documentation_tool",

    # Dependencies
    "DependencyTool": "victor.tools.dependency_tool",
    "DependencyGraphTool": "victor.tools.dependency_graph",

    # CI/CD & Infrastructure
    "CICDTool": "victor.tools.cicd_tool",
    "PipelineTool": "victor.tools.pipeline_tool",
    "IACScannerTool": "victor.tools.iac_scanner_tool",

    # Execution
    "CodeExecutorTool": "victor.tools.code_executor_tool",
    "BatchProcessorTool": "victor.tools.batch_processor_tool",
}


def __getattr__(name: str):
    """Lazy import for coding tools."""
    if name in _CODING_TOOL_MAPPINGS:
        import importlib
        try:
            module_path = _CODING_TOOL_MAPPINGS[name]
            module_name, _ = module_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, name)
        except (ImportError, AttributeError):
            pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_all_coding_tools():
    """Get all coding tool classes.

    Returns:
        Dict mapping tool names to tool classes
    """
    tools = {}
    for name in _CODING_TOOL_MAPPINGS:
        try:
            tools[name] = __getattr__(name)
        except AttributeError:
            pass
    return tools


__all__ = [
    "__version__",
    "get_all_coding_tools",
    # Tool classes (lazy loaded)
    "CodeSearchTool",
    "CodeIntelligenceTool",
    "LanguageAnalyzerTool",
    "GraphTool",
    "ArchitectureSummaryTool",
    "CodeReviewTool",
    "AuditTool",
    "MetricsTool",
    "FileEditorTool",
    "PatchTool",
    "MergeTool",
    "LSPTool",
    "DocumentationTool",
    "DependencyTool",
    "DependencyGraphTool",
    "CICDTool",
    "PipelineTool",
    "IACScannerTool",
    "CodeExecutorTool",
    "BatchProcessorTool",
]
