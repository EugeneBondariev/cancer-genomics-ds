PYTHON = python
export PYTHONUNBUFFERED := 1

.PHONY: help install install-dev freeze push typecheck lint test fetch dge embed classify survival annotate enrich predict

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
	@echo "  fetch            Download all raw data files from UCSC Xena / GDC"
	@echo "  annotate         Fetch Ensembl→symbol annotation via MyGene.info"
	@echo ""
	@echo "Model:"
	@echo "  dge              Run one-vs-rest differential expression"
	@echo "  embed            PCA + UMAP dimensionality reduction"
	@echo "  classify         Train PAM50 classifier (CV + full fit)"
	@echo "  survival         Merge survival data and save interim file"
	@echo "  enrich           Run GSEA pathway enrichment for all subtypes"
	@echo "  predict          Select prognostic genes + fit Cox signature"

install:
	pip install -e .

install-dev: install
	pip install -r requirements-dev.txt

freeze:
	pip freeze > requirements.txt

push: lint typecheck test
	git push

typecheck:
	$(PYTHON) -m mypy src/

lint:
	$(PYTHON) -m ruff check . --fix

test:
	$(PYTHON) -m pytest

fetch:
	$(PYTHON) -m src.data.fetch_data

dge:
	$(PYTHON) -m src.features.differential_expression

embed:
	$(PYTHON) -m src.features.dimensionality_reduction

classify:
	$(PYTHON) -m src.models.classifier

survival:
	$(PYTHON) -m src.models.survival

annotate:
	$(PYTHON) -m src.data.annotation

enrich:
	$(PYTHON) -m src.features.pathway_enrichment

predict:
	$(PYTHON) -m src.models.survival_prediction
