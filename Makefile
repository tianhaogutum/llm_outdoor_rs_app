# Developer entry points. Run from the repo root.
#   make setup   create backend/.venv (if missing) and install backend/requirements-dev.txt
#   make seed    rebuild backend/data/app.db from the OSM catalogue (CATALOG=legacy: the 111-route seed)
#   make frontend  install and build the static frontend into frontend/out (served by the backend at /)
#   make test    run the backend test suite (LLM_PROVIDER=mock, no network)
#   make lint    ruff check on backend/
#   make dev     run the API with autoreload on http://localhost:8000
#   make data    rebuild the OSM route catalogue in backend/data/catalog (downloads ~0.9 GB once)
#   make data-stats  print every figure quoted in docs/DATA.md
#   make narratives  rebuild backend/data/narratives from the committed cache (no credentials);
#                    narratives-live: new route set / prompt (costs about 0.15 USD)
#   make llm-smoke  one live generate/embed/rank call per configured provider (costs a fraction of a cent)
#
# Evaluation harness (docs/EVAL.md). Recorded LLM calls replay from the committed frozen cache eval/cache/,
# so on a fresh clone these need no credentials unless something new has to be judged.
#   make personas      regenerate eval/personas/personas.jsonl (deterministic from cache)
#   make eval SYSTEM=v0_rules  run one system on all personas, judge pool holes, write eval/results/<name>.json
#   make eval-all      recompute every stored run against the current qrels (tables stay consistent)
#   make report        render docs/RESULTS.md from eval/results/*.json
#   make judge-pilot   judge pilot (candidates vs reference; cached)
#   make owner-set     blind 150-pair labelling set + interim second judge (eval/owner_labels/)
#   make kappa         owner labels vs judge (reports "owner labels missing" until the file exists)
#   make cg-tune       Phase 3b dev experiments (BM25 variants, embedded document texts, RRF k, precision)
#   make rank-pilot    Phase 3c ranker-API pilot on dev (which reranker, which document text)
#   make ltr-sample    judge a deeper random sample of dev candidates for the learned ranker (costs money)
#   make ltr           train the learned ranker on dev (LightGBM), write the model artefacts and _ltr.json
#   make mmr-sweep     Phase 3d MMR lambda sweep on dev (+ docs/mmr_tradeoff.png)
#   make latency       warm end-to-end latency benchmark of the default system (live calls, cents)
#   make ltr-robust    retrain the default ranker without length_cap_ratio / the suspicious group (dev only)
#   make simulate      Phase 5 feedback simulation: 10 rounds per test persona (+ docs/convergence.png)
#   make sim-tune      Phase 5 feedback-parameter grid on a dev subset (judges holes: costs money)
#   make chat-eval     scripted chat slot evaluation, replayed from eval/chat/cache (no provider call)
#   make freeze        export every touched LLM cache entry to eval/cache/
#   make replay-check  clean cache + cache-only + no credentials: reproduce personas, judgements, results
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

CATALOG     ?= osm
CATALOG_DIR ?= $(abspath $(BACKEND)/data/catalog)

SYSTEM ?= v0_rules

# LightGBM needs an OpenMP runtime. On macOS Homebrew's libomp is not assumed; scikit-learn's wheel ships
# libomp.dylib. DYLD_* variables are stripped when make starts /bin/sh, so the variable is set inline in
# the recipe (LGBM_ENV) rather than exported. On Linux (CI) it is empty.
ifeq ($(shell uname -s),Darwin)
LGBM_ENV := DYLD_FALLBACK_LIBRARY_PATH=$(shell $(PYTHON) -c "import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), '.dylibs'))" 2>/dev/null)
else
LGBM_ENV :=
endif

.PHONY: setup seed frontend test lint dev llm-smoke data data-stats personas eval eval-all report judge-pilot owner-set \
	kappa freeze replay-check cg-tune rank-pilot ltr-sample ltr mmr-sweep latency ltr-robust simulate sim-tune \
	chat-eval
.PHONY: narratives narratives-live narratives-check

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
ifeq ($(CATALOG),legacy)
	cd $(BACKEND) && $(PYTHON) scripts/seed_from_jsonl.py --file $(SEED_FILE) --db $(DB)
else ifeq ($(CATALOG),osm)
	cd $(BACKEND) && $(PYTHON) scripts/seed_from_catalog.py --catalog $(CATALOG_DIR) --db $(DB)
else
	$(error CATALOG must be osm or legacy, not '$(CATALOG)')
endif

frontend:
	cd frontend && pnpm install --frozen-lockfile && pnpm build

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

personas:
	cd $(BACKEND) && $(PYTHON) -m evalkit.personas

eval:
	cd $(BACKEND) && $(PYTHON) -m evalkit.evaluate --system $(SYSTEM)

eval-all:
	cd $(BACKEND) && $(PYTHON) -m evalkit.evaluate --all

report:
	cd $(BACKEND) && $(PYTHON) -m evalkit.report

judge-pilot:
	cd $(BACKEND) && $(PYTHON) -m evalkit.pilot

owner-set:
	cd $(BACKEND) && $(PYTHON) -m evalkit.owner build && $(PYTHON) -m evalkit.owner second

kappa:
	cd $(BACKEND) && $(PYTHON) -m evalkit.owner kappa

cg-tune:
	cd $(BACKEND) && $(PYTHON) -m evalkit.cg tune

rank-pilot:
	cd $(BACKEND) && $(PYTHON) -m evalkit.rank_pilot

ltr-sample:
	cd $(BACKEND) && $(PYTHON) -m evalkit.ltr sample

ltr:
	cd $(BACKEND) && $(LGBM_ENV) $(PYTHON) -m evalkit.ltr train

mmr-sweep:
	cd $(BACKEND) && $(PYTHON) -m evalkit.mmr_sweep

latency:
	cd $(BACKEND) && $(PYTHON) -m evalkit.latency

ltr-robust:
	cd $(BACKEND) && $(LGBM_ENV) $(PYTHON) -m evalkit.ltr train-robust

simulate:
	cd $(BACKEND) && $(PYTHON) -m evalkit.simulate run

sim-tune:
	cd $(BACKEND) && $(PYTHON) -m evalkit.simulate tune

chat-eval:
	cd $(BACKEND) && $(PYTHON) -m evalkit.chat.run --all --cache-only

freeze:
	cd $(BACKEND) && $(PYTHON) -m evalkit.freeze export

replay-check:
	cd $(BACKEND) && $(PYTHON) -m evalkit.freeze verify

narratives:
	cd $(BACKEND) && $(PYTHON) scripts/narratives.py generate && $(PYTHON) scripts/narratives.py check

narratives-live:
	cd $(BACKEND) && $(PYTHON) scripts/narratives.py select && $(PYTHON) scripts/narratives.py generate --live --budget 0.5

narratives-check:
	cd $(BACKEND) && $(PYTHON) scripts/narratives.py check
