# ─── Config ───────────────────────────────────────────────────────────────────
PYTHON     := /Users/lavi./opt/anaconda3/envs/mcp-rag/bin/python
STREAMLIT  := /Users/lavi./opt/anaconda3/envs/mcp-rag/bin/streamlit
PORT       := 8501

.DEFAULT_GOAL := help

# ─── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  PDF RAG MCP Server — available targets"
	@echo "  ────────────────────────────────────────"
	@echo "  make install        Install all dependencies (editable)"
	@echo "  make install-dev    Install with dev extras (ruff, pytest)"
	@echo "  make ingest         Parse PDFs and populate the vector store"
	@echo "  make ingest-clean   Wipe and rebuild the vector store from scratch"
	@echo "  make server         Start the MCP server (stdio transport)"
	@echo "  make ui             Launch the Streamlit UI on port $(PORT)"
	@echo "  make test           Run the test suite"
	@echo "  make lint           Run ruff linter"
	@echo "  make fmt            Auto-format with ruff"
	@echo "  make clean          Remove __pycache__ and .ruff_cache"
	@echo ""

# ─── Install ──────────────────────────────────────────────────────────────────
.PHONY: install
install:
	$(PYTHON) -m pip install -e .

.PHONY: install-dev
install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

# ─── Data pipeline ────────────────────────────────────────────────────────────
.PHONY: ingest
ingest:
	$(PYTHON) scripts/ingest.py

.PHONY: ingest-clean
ingest-clean:
	$(PYTHON) scripts/ingest.py --clear

# ─── Runtime ──────────────────────────────────────────────────────────────────
.PHONY: server
server:
	$(PYTHON) main.py

.PHONY: ui
ui:
	$(STREAMLIT) run app.py --server.port $(PORT)

# ─── Quality ──────────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: lint
lint:
	$(PYTHON) -m ruff check src/ scripts/ app.py main.py

.PHONY: fmt
fmt:
	$(PYTHON) -m ruff format src/ scripts/ app.py main.py
	$(PYTHON) -m ruff check --fix src/ scripts/ app.py main.py

# ─── Cleanup ──────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	@echo "Clean complete."
