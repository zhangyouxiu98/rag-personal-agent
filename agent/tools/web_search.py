"""Web search fallback tool using DuckDuckGo."""

from agent.core.tools import BaseTool


class WebSearchTool(BaseTool):
    """Fallback web search when local knowledge base has insufficient results.

    Uses DuckDuckGo for privacy-friendly, no-API-key-required web search.
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information not found in the local knowledge base"

    async def execute(self, query: str, num_results: int = 5, **kwargs) -> str:
        """Execute a web search and return formatted results.

        Args:
            query: The search query.
            num_results: Maximum number of results to return.

        Returns:
            Formatted search results as a string, or empty string on failure.
        """
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append(
                        f"- {r.get('title', 'Untitled')}\n"
                        f"  {r.get('body', '')[:300]}\n"
                        f"  URL: {r.get('href', '')}"
                    )

            if results:
                return "Web search results:\n\n" + "\n\n".join(results)
            return ""

        except ImportError:
            return "Web search unavailable: duckduckgo_search not installed."
        except Exception as e:
            return f"Web search failed: {e}"
