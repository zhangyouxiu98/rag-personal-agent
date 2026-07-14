"""Structured database query tool."""

import sqlite3
from pathlib import Path

from agent.core.tools import BaseTool


class DatabaseTool(BaseTool):
    """Query structured SQLite databases or CSV files.

    Useful for querying tabular data that is hard to embed semantically.
    """

    def __init__(self, db_path: str = None):
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def description(self) -> str:
        return "Query structured data sources (SQLite, CSV) for precise answers"

    async def execute(self, query: str, db_path: str = None, **kwargs) -> str:
        """Execute a SQL or CSV query.

        Args:
            query: SQL statement or search query for CSV files.
            db_path: Path to the database file.

        Returns:
            Query results as formatted text.
        """
        path = db_path or self._db_path
        if not path:
            return "No database path configured."

        p = Path(path)

        if not p.exists():
            return f"Database file not found: {path}"

        if p.suffix.lower() == ".csv":
            return await self._query_csv(p, query)
        else:
            return await self._query_sqlite(p, query)

    async def _query_sqlite(self, path: Path, sql: str) -> str:
        """Execute a SQL query against SQLite."""
        try:
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            cursor.execute(sql)

            # Check if it's a SELECT query
            if sql.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description] if cursor.description else []

                if not rows:
                    conn.close()
                    return "Query returned no results."

                # Format as table
                lines = [" | ".join(columns)]
                lines.append("-" * len(lines[0]))
                for row in rows[:20]:  # Limit to 20 rows
                    lines.append(" | ".join(str(v) for v in row))

                if len(rows) > 20:
                    lines.append(f"... ({len(rows) - 20} more rows)")

                conn.close()
                return "\n".join(lines)
            else:
                conn.commit()
                affected = cursor.rowcount
                conn.close()
                return f"Query executed. Rows affected: {affected}"
        except Exception as e:
            return f"Database query error: {e}"

    async def _query_csv(self, path: Path, query: str) -> str:
        """Search/filter a CSV file for relevant rows."""
        import csv

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return "CSV file is empty."

            # Simple keyword matching across all columns
            keywords = query.lower().split()
            matching = []
            for row in rows:
                row_text = " ".join(str(v).lower() for v in row.values())
                if all(kw in row_text for kw in keywords):
                    matching.append(row)

            if not matching:
                return "No matching rows found."

            # Format results
            columns = list(matching[0].keys())
            lines = [" | ".join(columns)]
            lines.append("-" * len(lines[0]))
            for row in matching[:10]:
                lines.append(" | ".join(str(row.get(c, "")) for c in columns))

            if len(matching) > 10:
                lines.append(f"... ({len(matching) - 10} more rows)")

            return "\n".join(lines)
        except Exception as e:
            return f"CSV query error: {e}"
