"""
Smoke-test a running server over HTTP (CI's docker job; docs/DEPLOY.md). Standard library only, so it runs
on the CI host without the backend's dependencies.

    python3 backend/scripts/smoke_container.py --base-url http://127.0.0.1:8000 [--expect-offline]

Walks the demo: health, the static page, serving status, a profile with home and intent, explained
recommendations, a chat turn, feedback, completing a route, guest recommendations. ``--expect-offline``
additionally asserts that nothing came from a live LLM (no credentials in the container): narratives are
pre-generated or template, the chat answered with the keyword rules, recommendations did not use an
embedding call. Prints one JSON summary; exit status 1 on the first failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


class Smoke:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.timings: dict[str, float] = {}

    def call(self, name: str, method: str, path: str, body: Any = None, expect: int = 200) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status, raw, ctype = resp.status, resp.read(), resp.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            status, raw, ctype = e.code, e.read(), e.headers.get("content-type", "")
        self.timings[name] = round((time.perf_counter() - t) * 1000, 1)
        check(status == expect, f"{name}: HTTP {status}, expected {expect}: {raw[:300]!r}")
        return json.loads(raw) if "json" in ctype else raw.decode("utf-8", "replace")


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL {msg}", file=sys.stderr)
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--expect-offline", action="store_true")
    args = ap.parse_args(argv)
    s = Smoke(args.base_url)
    out: dict[str, Any] = {}

    check(s.call("healthz", "GET", "/healthz") == {"status": "ok"}, "healthz body")
    html = s.call("static_index", "GET", "/")
    check("<html" in html.lower() and "_next/" in html, "static index is not the Next.js export")
    status = s.call("recsys_status", "GET", "/api/recsys/status")
    rs = status["recsys"]
    check(rs.get("ok") is True, f"pipeline not loaded: {rs}")
    check(rs.get("db", {}).get("ok", True) is not False, f"DB/catalogue check failed: {rs.get('db')}")
    out["recsys"] = {k: rs.get(k) for k in ("configured", "catalogue_routes", "total_ms", "peak_rss_mb_after")}
    out["rate_limit_enabled"] = status["rate_limit"]["enabled"]
    out["daily_budget_cap_usd"] = status["daily_budget"]["cap_usd"]

    profile = s.call("create_profile", "POST", "/api/profiles", {
        "fitness": "intermediate", "type": ["natural-scenery"], "narrative": "mystery",
        "home": {"name": "Grainau"}, "travel_radius_km": 40, "distance_cap_km": 15,
        "intent": "easy lake loop with a hut"}, expect=201)
    pid = profile["id"]
    out["welcome_summary_source"] = profile.get("welcome_summary_source")
    check(bool(profile["welcome_summary"]), "empty welcome summary")

    recs = s.call("recommendations", "GET", f"/api/routes/recommendations?profile_id={pid}&category=hiking&limit=10")
    routes = recs["routes"]
    check(len(routes) == 10, f"expected 10 routes, got {len(routes)}")
    for r in routes:
        check(bool(r.get("reason")) and r.get("explanation") is not None, f"route {r.get('id')} without explanation")
        check(r["explanation"]["stages"]["filter"]["verdict"] == "pass", f"route {r['id']} did not pass the filter")
    sb = recs["served_by"]
    out["served_by"] = {k: sb.get(k) for k in ("configured_system", "system", "path", "reason", "degraded",
                                                "latency_ms")}
    out["first_reason"] = routes[0]["reason"]

    turn = s.call("chat_turn", "POST", f"/api/profiles/{pid}/chat", {"message": "a cycling tour along a river"})
    out["chat_model"] = turn.get("model")
    check(turn["slots"]["activity"] == "cycling", f"chat did not set the activity: {turn['slots']}")

    s.call("feedback", "POST", f"/api/profiles/{pid}/feedback",
           {"route_id": routes[-1]["id"], "reason": "too-far"}, expect=201)
    done = s.call("complete_route", "POST", f"/api/profiles/{pid}/souvenirs",
                  {"route_id": routes[0]["id"], "completed_quest_ids": []})
    out["summary_source"] = done.get("summary_source")
    check(bool(done["souvenir"]["genai_summary"]) and bool(done["souvenir"]["pixel_image_svg"]), "souvenir incomplete")

    guest = s.call("guest_recommendations", "GET", "/api/routes/recommendations?category=hiking&limit=5")
    check(len(guest["routes"]) == 5, "guest recommendations")

    if args.expect_offline:
        offline = ("pregenerated", "template")
        check(out["welcome_summary_source"] in offline, f"welcome from {out['welcome_summary_source']}")
        check(out["summary_source"] in ("pregenerated", "template"), f"summary from {out['summary_source']}")
        check(out["chat_model"] == "keyword-rules", f"chat answered by {out['chat_model']}")
        check(sb["path"] in ("no_llm", "pipeline") and (sb["path"] == "no_llm" or sb["reason"] == "no_intent"),
              f"recommendations used a provider: {sb}")
    out["timings_ms"] = s.timings
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
