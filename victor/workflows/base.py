
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# Forward-reference ToolRegistry to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from victor.tools.base import ToolRegistry


class BaseWorkflow(ABC):
    """
    Abstract base class for all workflows.
    A workflow is a sequence of steps that use tools to achieve a higher-level goal.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the workflow."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A short description of what the workflow does."""
        pass

    @abstractmethod
    async def run(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Executes the workflow.

        Args:
            context: The context containing shared resources like the ToolRegistry.
            **kwargs: Arguments specific to the workflow.

        Returns:
            A dictionary containing the results of the workflow execution.
        """
        pass


class WorkflowRegistry:
    """A registry for discovering and managing available workflows."""

    def __init__(self):
        self._workflows: Dict[str, BaseWorkflow] = {}

    def register(self, workflow: BaseWorkflow):
        """Registers a workflow instance."""
        if workflow.name in self._workflows:
            raise ValueError(f"Workflow '{workflow.name}' is already registered.")
        self._workflows[workflow.name] = workflow

    def get(self, name: str) -> Optional[BaseWorkflow]:
        """Retrieves a workflow by its name."""
        return self._workflows.get(name)

    def list_workflows(self) -> list[BaseWorkflow]:
        """Returns a list of all registered workflows."""
        return list(self._workflows.values())

