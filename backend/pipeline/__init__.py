"""Phase 1 data pipeline: OSM route relations + SRTM elevation -> route catalogue.

Entry points:
    python -m pipeline.build   # download (cached), extract, enrich, write backend/data/catalog/
    python -m pipeline.stats   # print every figure quoted in docs/DATA.md

Everything is deterministic: no randomness, no network calls other than the
cached downloads of the pinned input files, no LLM.
"""

PIPELINE_VERSION = "1.0.0"
