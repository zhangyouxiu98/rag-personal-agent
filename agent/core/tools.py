"""Tool abstraction and registry for agent-callable tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """Abstract base for all tools the agent can invoke."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used for registration and invocation."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with the given arguments.

        Returns:
            String result from the tool execution.
        """
        ...

    def to_schema(self) -> Dict[str, Any]:
        """Return a JSON-schema-like description for LLM tool-use."""
        return {
            "name": self.name,
            "description": self.description,
        }


class ToolRegistry:
    """Central registry that manages all available tools.

    Tools are registered by name and can be retrieved for execution
    or listed for LLM tool-use prompts.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool instance."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        """Get a tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools with names and descriptions."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs) -> str:
        """Execute a registered tool by name."""
        tool = self.get(name)
        return await tool.execute(**kwargs)

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)
