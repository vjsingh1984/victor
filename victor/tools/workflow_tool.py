
from typing import Any, Dict

from victor.tools.decorators import tool
from victor.workflows.base import WorkflowRegistry


@tool
async def run_workflow(
    workflow_name: str, context: dict, workflow_args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs a pre-defined, multi-step workflow to automate a complex task.

    Args:
        workflow_name: The name of the workflow to run (e.g., 'new_feature').
        context: The tool context provided by the orchestrator.
        workflow_args: A dictionary of arguments required by the specific workflow.

    Returns:
        A dictionary containing the results of the workflow execution.
    """
    workflow_registry: WorkflowRegistry = context.get("workflow_registry")
    if not workflow_registry:
        return {"error": "WorkflowRegistry not found in context."}

    workflow = workflow_registry.get(workflow_name)
    if not workflow:
        return {"error": f"Workflow '{workflow_name}' not found."}

    try:
        # Pass the full context down to the workflow
        result = await workflow.run(context, **workflow_args)
        return result
    except Exception as e:
        return {"error": f"An unexpected error occurred while running workflow '{workflow_name}': {e}"}

