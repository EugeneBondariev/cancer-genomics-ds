PYTHON = python
export PYTHONUNBUFFERED := 1

.PHONY: help install install-dev freeze push typecheck lint test fetch dge

help:
	@echo ""
	@echo "Setup:"
	@echo "  install          Install runtime deps + package (editable)"
	@echo "  install-dev      Also install dev tooling (mypy, stubs)"
	@echo "  freeze           Snapshot environment to requirements.txt"
	@echo ""
	@echo "Quality:"
	@echo "  push             Run typecheck + test, then git push"
	@echo "  typecheck        Run mypy over src/"
	@echo "  lint             Run ruff linter"
	@echo "  test             Run the test suite"
	@echo ""
	@echo "Data:"
	@echo "  fetch            Download all raw data files from UCSC Xena"
	@echo ""
	@echo "Model:"
	@echo "  dge              Run one-vs-rest differential expression"

install:
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt

freeze:
	pip freeze > requirements.txt

push: typecheck test
	git push

typecheck:
	$(PYTHON) -m mypy src/

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

fetch:
	$(PYTHON) -m src.data.fetch_data

dge:
	$(PYTHON) -m src.features.differential_expression
