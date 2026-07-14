"""Generic HTTP API calling tool."""

from agent.core.tools import BaseTool


class APITool(BaseTool):
    """Generic REST API caller for external services.

    Supports GET, POST, PUT, and DELETE methods with headers and body.
    """

    @property
    def name(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return "Call external REST APIs to fetch or send data"

    async def execute(
        self,
        url: str,
        method: str = "GET",
        params: dict = None,
        headers: dict = None,
        body: dict = None,
        **kwargs,
    ) -> str:
        """Call an external REST API.

        Args:
            url: Full URL to call.
            method: HTTP method (GET, POST, PUT, DELETE).
            params: URL query parameters.
            headers: HTTP headers.
            body: JSON request body (for POST/PUT).

        Returns:
            Response body as text, or error message on failure.
        """
        try:
            import httpx

            method = method.upper()
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method == "GET":
                    resp = await client.get(url, params=params, headers=headers)
                elif method == "POST":
                    resp = await client.post(url, json=body, params=params, headers=headers)
                elif method == "PUT":
                    resp = await client.put(url, json=body, params=params, headers=headers)
                elif method == "DELETE":
                    resp = await client.delete(url, params=params, headers=headers)
                else:
                    return f"Unsupported HTTP method: {method}"

                resp.raise_for_status()

                # Try to format JSON nicely
                try:
                    import json
                    data = resp.json()
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except Exception:
                    return resp.text[:2000]  # Truncate long responses

        except ImportError:
            return "API tool unavailable: httpx not installed."
        except Exception as e:
            return f"API call failed: {e}"
