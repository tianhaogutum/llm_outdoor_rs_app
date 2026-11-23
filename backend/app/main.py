import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Scope

from .api.v1 import achievements, chat, logs, places, profiles, routes, souvenirs
from .database import close_db, get_db, init_db
from .llm_logger import _llm_messages, get_recent_messages
from .logger import get_logger, init_logging_from_settings
from .models.entities import DemoProfile, Route
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    settings = get_settings()

    # Initialize logging system
    init_logging_from_settings()
    # httpx logs every request URL at INFO; Vertex URLs contain the Google Cloud project id, which must not
    # reach the logs (docs/DEPLOY.md). Warnings and errors still show.
    for noisy in ("httpx", "httpx2", "httpcore"):  # httpx2: the client google-genai uses
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logger = get_logger(__name__)
    logger.info("🚀 Application starting...")

    init_db(settings)
    logger.info("✅ Database initialization completed")

    # Seed achievements on startup
    try:
        from app.database import get_db_session
        from app.services.achievement_service import seed_achievements
        logger.info("🌱 Seeding achievement data...")
        async with await get_db_session() as session:
            await seed_achievements(session)
        logger.info("✅ Achievement data seeding completed")
    except Exception as e:
        # Log but don't fail startup if seeding fails
        logger.warning(f"⚠️ Achievement data seeding failed: {e}", exc_info=True)

    # Recommender resources: catalogue, BM25 index, document matrix, ranker trees (once per process)
    from app.services import recsys_service

    t0 = time.perf_counter()
    res = await asyncio.to_thread(recsys_service.load_resources, settings)
    await recsys_service.warm_up(res)
    res.startup["lifespan_recsys_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    app.state.recsys = res
    try:  # DB vs catalogue (the request path re-checks per database URL)
        from app.database import get_db_session

        if res.catalog is not None and settings.recsys_system != "v0_rules":
            async with await get_db_session() as session:
                await recsys_service.check_db(session, res)
    except Exception as e:  # e.g. DB not migrated yet; the first request checks again
        logger.warning(f"⚠️ recsys DB check at start-up failed: {e}")
    logger.info(f"🧭 recsys ready: {json.dumps(recsys_service.startup_report(res), default=str)}")

    logger.info("🎉 Application startup completed")
    yield

    # Shutdown
    logger.info("🛑 Application shutting down...")
    await close_db()
    logger.info("✅ Database connections closed")


# The Next.js static export (`cd frontend && pnpm build` writes frontend/out). Override with
# FRONTEND_DIST_DIR; when the directory has no index.html, `/` falls back to the API dashboard.
DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "out"

# Paths that belong to the API even when nothing under them matches: answer them with a JSON 404
# instead of the frontend's HTML 404 page.
API_PREFIXES = ("api/", "docs", "redoc", "openapi.json", "healthz", "dashboard")


def frontend_dist_dir() -> Path:
    return Path(os.environ.get("FRONTEND_DIST_DIR") or DEFAULT_FRONTEND_DIST)


class FrontendStaticFiles(StaticFiles):
    """Serves the exported frontend at `/` without shadowing the API (routes are matched first)."""

    async def get_response(self, path: str, scope: Scope):
        if path == "api" or path.startswith(API_PREFIXES):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return await super().get_response(path, scope)


def create_app() -> FastAPI:
    """
    Application factory that configures FastAPI along with shared dependencies.
    """
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # Next.js default dev port
            "http://localhost:3001",  # Alternative port
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],  # Explicitly allow OPTIONS
        allow_headers=["*"],  # Allow all headers
        expose_headers=["*"],
    )

    @app.get("/dashboard", tags=["info"], response_class=HTMLResponse)
    async def dashboard(db: AsyncSession = Depends(get_db)):
        """API dashboard (statistics, endpoint links, live LLM stream). Served at `/` when no frontend is built."""
        # Get statistics from database
        stats = {}
        try:
            # Count users
            user_count_result = await db.execute(select(func.count(DemoProfile.id)))
            stats["total_users"] = user_count_result.scalar() or 0

            # Count routes
            route_count_result = await db.execute(select(func.count(Route.id)))
            stats["total_routes"] = route_count_result.scalar() or 0

            # Calculate total XP
            total_xp_result = await db.execute(select(func.sum(DemoProfile.total_xp)))
            stats["total_xp_earned"] = total_xp_result.scalar() or 0

            # Get average level
            avg_level_result = await db.execute(select(func.avg(DemoProfile.level)))
            avg_level = avg_level_result.scalar()
            stats["average_level"] = round(float(avg_level), 2) if avg_level else 0.0
        except Exception as e:
            stats = {
                "total_users": 0,
                "total_routes": 0,
                "total_xp_earned": 0,
                "average_level": 0.0,
                "error": str(e)
            }

        # Determine database type
        db_type = "SQLite"
        if settings.database_url.startswith("postgresql"):
            db_type = "PostgreSQL"

        # Load HTML template
        template_path = Path(__file__).parent / "templates" / "dashboard.html"
        if template_path.exists():
            html_content = template_path.read_text(encoding="utf-8")
        else:
            # Fallback HTML if template not found
            html_content = """
            <!DOCTYPE html>
            <html>
            <head><title>TrailSaga – Hogwarts Expedition Series API</title></head>
            <body>
                <h1>TrailSaga – Hogwarts Expedition Series Backend API</h1>
                <p>Template file not found. Please check templates/dashboard.html</p>
            </body>
            </html>
            """

        # Replace template variables
        html_content = html_content.replace("{{ version }}", settings.version)
        html_content = html_content.replace("{{ status }}", "🟢 Operational")
        html_content = html_content.replace("{{ description }}", "Gamified outdoor adventure platform with AI-powered storytelling")  # noqa: E501
        html_content = html_content.replace("{{ statistics.total_users }}", str(stats.get("total_users", 0)))
        html_content = html_content.replace("{{ statistics.total_routes }}", str(stats.get("total_routes", 0)))
        html_content = html_content.replace("{{ statistics.total_xp_earned }}", str(stats.get("total_xp_earned", 0)))
        html_content = html_content.replace("{{ statistics.average_level }}", str(stats.get("average_level", 0.0)))
        html_content = html_content.replace("{{ database.type }}", db_type)
        html_content = html_content.replace("{{ database.status }}", "✅ Connected")
        html_content = html_content.replace("{{ llm.service }}", "Ollama")
        html_content = html_content.replace("{{ llm.model }}", settings.ollama_model)

        return HTMLResponse(content=html_content)

    @app.get("/healthz", tags=["health"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/llm-stream", tags=["monitoring"])
    async def llm_stream():
        """Server-Sent Events endpoint for real-time LLM output streaming."""
        async def event_generator():
            last_count = 0
            while True:
                current_count = len(_llm_messages)
                if current_count > last_count:
                    # New messages available, send them
                    new_messages = list(_llm_messages)[last_count:]
                    for message in new_messages:
                        yield f"data: {message.to_json()}\n\n"
                    last_count = current_count
                else:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"

                await asyncio.sleep(0.5)  # Check every 500ms

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @app.get("/api/recsys/status", tags=["monitoring"])
    async def recsys_status():
        """Serving status: configured system, start-up timings and memory, DB check, daily budget."""
        from app.services import recsys_service
        from app.services.demo_guard import get_guard

        res = getattr(app.state, "recsys", None)
        return {
            "recsys": recsys_service.startup_report(res),
            "daily_budget": get_guard().budget.status(),
            "rate_limit": {"enabled": settings.rate_limit_enabled, "window_s": settings.rate_limit_window_s,
                           "limits": get_guard().limiter.limits},
        }

    @app.get("/api/llm-messages", tags=["monitoring"])
    async def get_llm_messages(limit: int = 20):
        """Get recent LLM messages."""
        return {"messages": get_recent_messages(limit)}

    app.include_router(profiles.router, prefix="/api")
    app.include_router(routes.router, prefix="/api")
    app.include_router(souvenirs.router, prefix="/api")
    app.include_router(achievements.router, prefix="/api")
    app.include_router(logs.router, prefix="/api")
    app.include_router(places.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")

    dist = frontend_dist_dir()
    if (dist / "index.html").is_file():
        # Mounted last: every API route above is matched before the static files.
        app.mount("/", FrontendStaticFiles(directory=dist, html=True), name="frontend")
    else:
        app.add_api_route("/", dashboard, methods=["GET"], response_class=HTMLResponse, include_in_schema=False)

    return app


app = create_app()

