"""Advanced workflow examples."""

import asyncio
import tempfile
import os
from victor import Agent


async def document_processing_pipeline(content: str):
    """Process document through multiple stages."""

    # Create workflow YAML
    workflow_yaml = """
name: "Document Processor"
description: "Multi-stage document processing"

nodes:
  - id: "extract_info"
    type: "agent"
    config:
      prompt: |
        Extract key information from this document:
        {{content}}

        Identify:
        1. Main topics
        2. Key entities (people, organizations, dates)
        3. Important numbers
        4. Action items

  - id: "summarize"
    type: "agent"
    config:
      prompt: |
        Create an executive summary based on:
        {{extract_info.output}}

  - id: "translate"
    type: "agent"
    config:
      prompt: |
        Translate to Spanish:
        {{summarize.output}}

  - id: "format"
    type: "agent"
    config:
      prompt: |
        Format as a structured report with sections:
        - Executive Summary
        - Key Findings
        - Translation
        {{translate.output}}

edges:
  - from: "start"
    to: "extract_info"
  - from: "extract_info"
    to: "summarize"
  - from: "summarize"
    to: "translate"
  - from: "translate"
    to: "format"
  - from: "format"
    to: "complete"
"""

    # Save workflow
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        # Run workflow
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"content": content}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def code_review_workflow(code: str, file_path: str = "example.py"):
    """Automated code review workflow."""

    workflow_yaml = f"""
name: "Code Review Workflow"
description: "Automated code review and analysis"

nodes:
  - id: "check_style"
    type: "agent"
    config:
      tools: ["read"]
      vertical: "coding"
      prompt: |
        Review this code for style (PEP 8):
        {code}

        Check:
        - Line length
        - Naming conventions
        - Import ordering
        - Spacing and indentation

  - id: "check_security"
    type: "agent"
    config:
      tools: ["read"]
      vertical: "security"
      prompt: |
        Review this code for security issues:
        {code}

        Check:
        - SQL injection
        - XSS vulnerabilities
        - Hardcoded secrets
        - Input validation

  - id: "check_quality"
    type: "agent"
    config:
      tools: ["read"]
      vertical: "coding"
      prompt: |
        Review this code for quality:
        {code}

        Check:
        - Error handling
        - Edge cases
        - Type hints
        - Documentation
        - Test coverage

  - id: "suggest_improvements"
    type: "agent"
    config:
      tools: ["read"]
      vertical: "coding"
      prompt: |
        Based on these reviews:
        Style: {{check_style.output}}
        Security: {{check_security.output}}
        Quality: {{check_quality.output}}

        Suggest prioritized improvements with:
        1. Severity level
        2. Specific changes
        3. Code examples
        4. Priority order

edges:
  - from: "start"
    to: "check_style"
  - from: "start"
    to: "check_security"
  - from: "start"
    to: "check_quality"
  - from: "check_style"
    to: "suggest_improvements"
  - from: "check_security"
    to: "suggest_improvements"
  - from: "check_quality"
    to: "suggest_improvements"
  - from: "suggest_improvements"
    to: "complete"
"""

    # Save and run workflow
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(workflow_path, input={})
        return result.content
    finally:
        os.unlink(workflow_path)


async def research_workflow(topic: str):
    """Comprehensive research workflow."""

    workflow_yaml = f"""
name: "Research Workflow"
description: "Comprehensive research on a topic"

nodes:
  - id: "define_scope"
    type: "agent"
    config:
      tools: ["web_search"]
      prompt: |
        Define the research scope for: {topic}

        Identify:
        1. Key aspects to research
        2. Subtopics to explore
        3. Types of sources needed
        4. Research questions

  - id: "gather_sources"
    type: "agent"
    config:
      tools: ["web_search", "web_fetch"]
      prompt: |
        Based on scope:
        {{define_scope.output}}

        Gather relevant sources:
        1. Academic papers
        2. Industry reports
        3. News articles
        4. Documentation

  - id: "analyze_sources"
    type: "agent"
    config:
      prompt: |
        Analyze the gathered sources:
        {{gather_sources.output}}

        For each source:
        1. Credibility assessment
        2. Key findings
        3. Limitations
        4. Relevance score

  - id: "synthesize"
    type: "agent"
    config:
      prompt: |
        Synthesize all research into comprehensive report:

        Topic: {topic}
        Scope: {{define_scope.output}}
        Sources: {{analyze_sources.output}}

        Include:
        1. Executive summary
        2. Background
        3. Key findings
        4. Methodology
        5. Sources and references

  - id: "create_bibliography"
    type: "agent"
    config:
      prompt: |
        Create bibliography in APA format from:
        {{gather_sources.output}}

edges:
  - from: "start"
    to: "define_scope"
  - from: "define_scope"
    to: "gather_sources"
  - from: "gather_sources"
    to: "analyze_sources"
  - from: "analyze_sources"
    to: "synthesize"
  - from: "synthesize"
    to: "create_bibliography"
  - from: "create_bibliography"
    to: "complete"
"""

    # Save and run
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create(vertical="research")
        result = await agent.run_workflow(workflow_path, input={})
        return result.content
    finally:
        os.unlink(workflow_path)


async def main():
    """Run advanced workflow examples."""
    print("=== Advanced Workflow Examples ===\n")

    # Example 1: Document processing
    print("1. Document Processing Pipeline:")
    content = """
    Victor AI Framework v0.5.8 was released on February 26, 2026.
    Key features include multi-provider support, workflow orchestration,
    and 33+ built-in tools. The framework is licensed under Apache 2.0.
    """
    doc_result = await document_processing_pipeline(content)
    print(doc_result[:400] + "...\n")

    # Example 2: Code review
    print("\n2. Code Review Workflow:")
    code = """
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
"""
    review_result = await code_review_workflow(code)
    print(review_result[:400] + "...\n")


if __name__ == "__main__":
    asyncio.run(main())
