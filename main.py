"""Root-level convenience entry point for the MCP server."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is discoverable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run()
