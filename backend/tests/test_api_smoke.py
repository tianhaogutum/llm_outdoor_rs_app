"""End-to-end smoke test of the API on the seeded test DB (LLM_PROVIDER=mock, no network)."""

from __future__ import annotations

import json

import pytest

from tests.conftest import GOLDEN_FILE


@pytest.fixture
def client(seeded_db, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.v1 import profiles
    from app.main import app

    async def no_llm_welcome(questionnaire):  # keep the test offline whatever genai_service does
        return "Welcome (test stub)"

    monkeypatch.setattr(profiles, "generate_welcome_summary", no_llm_welcome)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_profile_recommendations_and_feedback(client):
    answers = {"fitness": "beginner", "type": ["natural-scenery"], "narrative": "mystery"}
    resp = client.post("/api/profiles", json=answers)
    assert resp.status_code == 201, resp.text
    profile_id = resp.json()["id"]

    resp = client.get("/api/routes/recommendations", params={"profile_id": profile_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_personalized"] is True and body["total"] == 20

    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    case = next(
        c for c in golden["cases"]
        if (c["fitness"], c["adventure_type"], c["category"]) == ("beginner", "natural-scenery", None)
    )
    assert [r["id"] for r in body["routes"]] == [row[0] for row in case["ranking"][:20]]
    assert [r["recommendation_score"] for r in body["routes"]] == [row[1] for row in case["ranking"][:20]]

    top = body["routes"][0]["id"]
    resp = client.post(f"/api/profiles/{profile_id}/feedback", json={"route_id": top, "reason": "too-hard"})
    assert resp.status_code == 201, resp.text

    resp = client.get("/api/routes/recommendations", params={"profile_id": profile_id, "limit": 200})
    scores = {r["id"]: r["recommendation_score"] for r in resp.json()["routes"]}
    assert scores[top] < case["ranking"][0][1]
