"""Automation workflow recipes.

These workflows automate common tasks like content processing,
reporting, and batch operations.
"""

RECIPE_CATEGORY = "workflows/automation"
RECIPE_DIFFICULTY = "intermediate"
RECIPE_TIME = "20 minutes"


async def document_processing_pipeline():
    """Automated document processing pipeline.

    Processes documents through multiple stages:
    1. Extract information
    2. Summarize
    3. Categorize
    4. Generate report
    """
    import tempfile
    import os
    from victor import Agent

    workflow_yaml = """
name: "Document Processing Pipeline"
description: "Multi-stage document processing"

nodes:
  - id: "extract_info"
    type: "agent"
    config:
      prompt: |
        Extract key information from document:
        {{input.document}}

        Provide:
        1. Main topics
        2. Key entities
        3. Important dates/numbers
        4. Action items

  - id: "summarize"
    type: "agent"
    config:
      prompt: |
        Create executive summary:
        {{extract_info.output}}

  - id: "categorize"
    type: "agent"
    config:
      prompt: |
        Categorize document:
        {{extract_info.output}}

        Categories:
        - Technical (code, architecture, devops)
        - Business (strategy, finance, marketing)
        - Legal (compliance, contracts, IP)
        - HR (policies, procedures)

        Output primary and secondary categories.

  - id: "generate_report"
    type: "agent"
    config:
      prompt: |
        Generate comprehensive report:
        Summary: {{summarize.output}}
        Categorization: {{categorize.output}}

        Include:
        - Executive Summary
        - Key Findings
        - Recommendations
        - Appendices

edges:
  - from: "start"
    to: "extract_info"
  - from: "extract_info"
    to: "summarize"
  - from: "summarize"
    to: "categorize"
  - from: "categorize"
    to: "generate_report"
  - from: "generate_report"
    to: "complete"
"""

    # Save workflow
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        # Example usage
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"document": "Victor AI Framework v0.5.8 release notes..."}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def content_scheduling():
    """Schedule and organize content production."""
    import tempfile
    import os
    from victor import Agent

    workflow_yaml = """
name: "Content Scheduling Workflow"
description: "Plan and schedule content production"

nodes:
  - id: "analyze_requirements"
    type: "agent"
    config:
      prompt: |
        Analyze content requirements:
        {{input}}

        Identify:
        1. Content types needed
        2. Target audience
        3. Key messages
        4. Desired tone
        5. Channels for distribution

  - id: "create_calendar"
    type: "agent"
    config:
      prompt: |
        Based on requirements:
        {{analyze_requirements.output}}

        Create a content calendar for 2 weeks.

        For each day:
        - Content pieces to produce
        - Channel to distribute on
        - Target audience segment
        - Priority level

  - id: "assign_tasks"
    type: "agent"
    config:
      prompt: |
        Based on calendar:
        {{create_calendar.output}}

        Assign tasks to team members:
        - Content Writer
        - Designer
        - SEO Specialist
        - Social Media Manager

        For each task:
        - Owner
        - Deadline
        - Dependencies
        - Deliverables

  - id: "create_checklist"
    type: "agent"
    config:
      prompt: |
        Create checklist for content production:
        {{assign_tasks.output}}

        Include:
        - Pre-publishing checks
        - Quality criteria
        - SEO checklist
        - Approval workflow

edges:
  - from: "start"
    to: "analyze_requirements"
  - from: "analyze_requirements"
    to: "create_calendar"
  - from: "create_calendar"
    to: "assign_tasks"
  - from: "assign_tasks"
    to: "create_checklist"
  - from: "create_checklist"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"input": "Need content for product launch next month"}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def automated_reporting():
    """Generate automated reports from data."""
    import tempfile
    import os
    from victor import Agent

    workflow_yaml = """
name: "Automated Reporting"
description: "Generate reports from data"

nodes:
  - id: "analyze_data"
    type: "handler"
    config:
      tool: "read"
      arguments:
        file_path: "{{input.data_file}}"

  - id: "calculate_metrics"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Analyze this data:
        {{analyze_data.result}}

        Calculate:
        - Key metrics (KPIs)
        - Trends over time
        - Anomalies
        - Insights

  - id: "create_visualizations"
    type: "agent"
    config:
      tools: ["python"]
      prompt: |
        Generate Python code for visualizations:
        {{calculate_metrics.output}}

        Include:
        - Line charts for trends
        - Bar charts for comparisons
        - Pie charts for distributions

  - id: "write_report"
    type: "agent"
    config:
      tools: ["write"]
      prompt: |
        Write comprehensive report:
        Data: {{analyze_data.result}}
        Metrics: {{calculate_metrics.output}}
        Visualizations: {{create_visualizations.output}}

        Structure:
        1. Executive Summary
        2. Methodology
        3. Findings
        4. Visualizations
        5. Recommendations

        Save to: {{input.output_file}}

edges:
  - from: "start"
    to: "analyze_data"
  - from: "analyze_data"
    to: "calculate_metrics"
  - from: "calculate_metrics"
    to: "create_visualizations"
  - from: "create_visualizations"
    to: "write_report"
  - from: "write_report"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create(vertical="dataanalysis")
        result = await agent.run_workflow(
            workflow_path,
            input={
                "data_file": "sales_data.csv",
                "output_file": "monthly_report.md"
            }
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def batch_processing():
    """Process multiple items in batch."""
    import asyncio
    from victor import Agent

    items = [
        "item1@example.com",
        "item2@example.com",
        "item3@example.com",
    ]

    agent = Agent.create(temperature=0.3)

    tasks = []
    for item in items:
        task = agent.run(f"Analyze {item} and classify as spam or not.")
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    return results


async def email_automation():
    """Automated email processing workflow."""
    import tempfile
    import os
    from victor import Agent

    workflow_yaml = """
name: "Email Automation Workflow"
description: "Automated email processing and routing"

nodes:
  - id: "classify_email"
    type: "agent"
    config:
      prompt: |
        Classify this email:
        {{input.email}}

        Categories:
        - Sales inquiry
        - Support request
        - Partnership
        - Other

        Output category and confidence level.

  - id: "extract_details"
    type: "agent"
    config:
      prompt: |
        Extract details from:
        {{classify_email.output}}

        Extract:
        - Sender name
        - Email address
        - Company (if any)
        - Phone (if any)
        - Urgency
        - Key topics

  - id: "route_email"
    type: "agent"
    config:
      prompt: |
        Route this email:
        Classification: {{classify_email.output}}
        Details: {{extract_details.output}}

        Determine appropriate recipient:
        - Sales team (for inquiries)
        - Support team (for issues)
        - Partnerships team (for partnerships)
        - General (for others)

        Output recipient name and email.

  - id: "draft_response"
    type: "agent"
    config:
      prompt: |
        Draft response email:
        Details: {{extract_details.output}}
        Classification: {{classify_email.output}}

        Create professional, personalized response.

edges:
  - from: "start"
    to: "classify_email"
  - from: "classify_email"
    to: "extract_details"
  - from: "extract_details"
    to: "route_email"
  - from: "route_email"
    to: "draft_response"
  - from: "draft_response"
    to: "complete"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(workflow_yaml)
        workflow_path = f.name

    try:
        agent = Agent.create()
        result = await agent.run_workflow(
            workflow_path,
            input={"email": "Subject: Partnership inquiry...\n\n..."}
        )
        return result.content
    finally:
        os.unlink(workflow_path)


async def demo_automation_workflows():
    """Demonstrate automation workflows."""
    print("=== Automation Workflow Recipes ===\n")

    print("1. Document Processing Pipeline:")
    result = await document_processing_pipeline()
    print(result[:400] + "...\n")

    print("\n2. Batch Processing:")
    results = await batch_processing()
    for i, result in enumerate(results):
        print(f"  Item {i+1}: {result.content[:100]}...\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_automation_workflows())
