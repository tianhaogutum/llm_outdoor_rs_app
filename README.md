# TUM Lab: Recommender System 

A full-stack application with FastAPI backend and Next.js frontend, featuring outdoor route recommendations with gamification and story generation.

## Tech Stack

**Backend:**

- FastAPI (Python 3.12)
- SQLAlchemy with async SQLite
- Alembic for database migrations

**Frontend:**

- Next.js 16 with React 19
- TypeScript
- Tailwind CSS

## Quick Start

### Prerequisites

- Python 3.12
- Node.js 18+ (pnpm or npm)
- GNU make (for the shortcuts below)

### 1. Backend Setup

```bash
make setup   # creates backend/.venv if missing, installs backend/requirements-dev.txt
make seed    # builds backend/data/app.db from backend/data/seed/routes.jsonl
```

The database is not committed; `make seed` deletes `backend/data/app.db`, runs `alembic upgrade head`
and loads the route catalogue (routes, breakpoints, mini quests) from the JSONL seed file.
User data (profiles, souvenirs, feedback) is not part of the seed.

### Make targets

| Target | What it does |
|---|---|
| `make setup` | create `backend/.venv` (uv if available, else `python3 -m venv`) and install dev requirements |
| `make seed` | rebuild `backend/data/app.db` from `backend/data/seed/routes.jsonl` |
| `make test` | run `pytest` in `backend/` with `LLM_PROVIDER=mock` (no network) |
| `make lint` | `ruff check` on `backend/` (config in `backend/pyproject.toml`) |
| `make dev` | run the API with autoreload on port 8000 |

The targets use `backend/.venv/bin/python` when it exists, otherwise `python` from `PATH` (as in CI).

### 2. Frontend Setup

```bash
cd frontend
pnpm install  # or npm install
pnpm run build  # or npm run build
```

### 3. Run the Application

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Access the application at:

- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/healthz

Or use the dev script:

```bash
./scripts/dev.sh
```

## Development Workflow

1. **Set up environment** (first time only)

   ```bash
   # Backend
   cd backend && python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   alembic upgrade head

   # Frontend
   cd ../frontend && pnpm install
   ```

2. **Develop backend**

   - Edit code in `backend/app/`
   - Create migrations: `alembic revision --autogenerate -m "description"`
   - Apply migrations: `alembic upgrade head`

3. **Develop frontend**

   ```bash
   cd frontend
   pnpm run dev  # Runs on http://localhost:3000
   ```

4. **Build and serve together**

   ```bash
   # Build frontend
   cd frontend && pnpm run build

   # Run backend (serves both API and frontend)
   cd ../backend && source venv/bin/activate
   uvicorn app.main:app --reload --port 8000
   ```

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/v1/     # API endpoints
│   │   ├── models/     # Database models
│   │   ├── services/   # Business logic
│   │   ├── database.py # DB connection
│   │   └── main.py     # FastAPI app
│   ├── alembic/        # Database migrations
│   ├── data/           # SQLite database
│   ├── scripts/        # Utility scripts
│   └── requirements.txt
├── frontend/
│   ├── app/            # Next.js app directory
│   ├── components/     # React components
│   └── out/            # Build output (generated)
└── scripts/
    └── dev.sh          # Development server script
```

## Database

- **Default location**: `backend/data/app.db`
- **Migrations**: Managed with Alembic
  ```bash
  alembic revision --autogenerate -m "description"
  alembic upgrade head
  alembic downgrade -1
  ```
- **Environment**: Copy `backend/env.example` to `backend/.env` to customize `DATABASE_URL`

## Scripts

- `backend/scripts/seed_from_jsonl.py` - Fill an empty, migrated DB from a JSONL route file (used by `make seed`)
- `backend/scripts/export_seed.py` - Export the routes of a DB to `backend/data/seed/routes.jsonl`
- `scripts/dev.sh` - Start development server

## Tests

```bash
make test
```

Tests live in `backend/tests/` and never call an LLM or the network. `backend/tests/golden/` pins the
full ranking of the rule-based recommender (`v0_rules`) as produced by the original scorer.

## Environment Variables

See `backend/env.example` for available configuration options. Create `backend/.env` to override defaults.
