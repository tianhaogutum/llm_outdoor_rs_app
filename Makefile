# Developer entry points. Run from the repo root.
#   make setup   create backend/.venv (if missing) and install backend/requirements-dev.txt
#   make seed    rebuild backend/data/app.db from backend/data/seed/routes.jsonl
#   make test    run the backend test suite (LLM_PROVIDER=mock, no network)
#   make lint    ruff check on backend/
#   make dev     run the API with autoreload on http://localhost:8000
#   make data    rebuild the OSM route catalogue in backend/data/catalog (downloads ~0.9 GB once)
#   make data-stats  print every figure quoted in docs/DATA.md
#   make llm-smoke  one live generate/embed/rank call per configured provider (costs a fraction of a cent)
#
# Uses backend/.venv/bin/python when it exists, otherwise `python` from PATH (CI).
# Override with `make test PYTHON=...`.

BACKEND := backend
VENV    := $(BACKEND)/.venv

ifneq ($(wildcard $(VENV)/bin/python),)
PYTHON ?= $(abspath $(VENV)/bin/python)
else
PYTHON ?= python
endif

DB        ?= $(abspath $(BACKEND)/data/app.db)
SEED_FILE ?= $(abspath $(BACKEND)/data/seed/routes.jsonl)
PORT      ?= 8000

.PHONY: setup seed test lint dev llm-smoke data data-stats

setup:
	@if [ ! -x $(VENV)/bin/python ]; then \
		if command -v uv >/dev/null 2>&1; then uv venv --python 3.12 $(VENV); \
		else python3 -m venv $(VENV); fi; \
	fi
	@if command -v uv >/dev/null 2>&1; then uv pip install -p $(VENV) -r $(BACKEND)/requirements-dev.txt; \
	else $(VENV)/bin/python -m pip install -r $(BACKEND)/requirements-dev.txt; fi

seed:
	rm -f $(DB)
	cd $(BACKEND) && DATABASE_URL=sqlite:///$(DB) $(PYTHON) -m alembic upgrade head
	cd $(BACKEND) && $(PYTHON) scripts/seed_from_jsonl.py --file $(SEED_FILE) --db $(DB)

test:
	cd $(BACKEND) && LLM_PROVIDER=mock $(PYTHON) -m pytest $(PYTEST_ARGS)

lint:
	cd $(BACKEND) && $(PYTHON) -m ruff check .

dev:
	cd $(BACKEND) && $(PYTHON) -m uvicorn app.main:app --reload --port $(PORT)

llm-smoke:
	cd $(BACKEND) && $(PYTHON) scripts/llm_smoke.py

data:
	cd $(BACKEND) && $(PYTHON) -m pipeline.build

data-stats:
	cd $(BACKEND) && $(PYTHON) -m pipeline.stats
