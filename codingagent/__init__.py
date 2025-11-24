"""
CodingAgent - A universal terminal-based coding agent supporting multiple LLM providers.

Supports frontier models (Claude, GPT, Gemini) and open-source models
(Ollama, LMStudio, vLLM) with unified tool calling integration.
"""

__version__ = "0.1.0"
__author__ = "Vijay Singh"
__license__ = "MIT"

from codingagent.agent.orchestrator import AgentOrchestrator
from codingagent.config.settings import Settings

__all__ = ["AgentOrchestrator", "Settings"]
