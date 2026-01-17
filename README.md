# TrailSaga – Hogwarts Expedition Series

A gamified outdoor adventure platform with AI-powered storytelling that transforms hiking routes into immersive narrative experiences. Users complete quests, earn XP, unlock achievements, and collect souvenirs while exploring real-world hiking trails.

## 🎯 Project Overview

TrailSaga combines outdoor adventure with gamification and AI-generated storytelling. The platform:
- Recommends hiking routes based on user preferences
- Generates personalized stories for each route using LLM (Ollama)
- Tracks user progress through breakpoints along routes
- Awards XP and levels based on route completion
- Unlocks achievements and collectible souvenirs
- Provides an interactive hiking simulation experience

## 🏗️ Architecture

### Tech Stack

**Backend:**
- **Framework**: FastAPI (Python 3.8+)
- **Database**: SQLite (with async SQLAlchemy)
- **Migrations**: Alembic
- **LLM**: Ollama (Llama3.1:8b model)
- **API**: RESTful API with async endpoints

**Frontend:**
- **Framework**: Next.js 16 (React 19)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI
- **State Management**: React Hooks

### Project Structure

```
Team_1/
├── backend/                 # FastAPI backend application
│   ├── app/
│   │   ├── api/            # API endpoints (v1)
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic services
│   │   ├── database.py     # Database configuration
│   │   ├── main.py         # FastAPI application entry point
│   │   └── settings.py     # Application settings
│   ├── alembic/            # Database migrations
│   ├── data/               # Database and data files
│   ├── scripts/            # Utility scripts
│   ├── requirements.txt    # Python dependencies
│   └── alembic.ini         # Alembic configuration
├── frontend/               # Next.js frontend application
│   ├── app/                # Next.js app directory
│   ├── components/         # React components
│   ├── lib/                # Utilities and API client
│   └── package.json       # Node.js dependencies
└── scripts/                # Development scripts
```

## 📋 Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.8+** (Python 3.10+ recommended)
- **Node.js 18+** and **npm** or **pnpm**
- **Ollama** installed and running locally
  - Download from: https://ollama.ai
  - Install the `llama3.1:8b` model: `ollama pull llama3.1:8b`
- **Git** for version control

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Team_1
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install
# or if using pnpm:
pnpm install
```

### 4. Environment Configuration

#### Backend Environment Variables

Create a `.env` file in the `backend/` directory:

```bash
cd backend
cp env.example .env
```

Edit `.env` with your configuration (optional, defaults are provided):

```env
# Database configuration (optional - defaults to sqlite+aiosqlite:///backend/data/app.db)
# DATABASE_URL=sqlite+aiosqlite:///path/to/your/database.db

# Ollama configuration (optional - defaults shown)
# OLLAMA_API_URL=http://127.0.0.1:11434/api/generate
# OLLAMA_MODEL=llama3.1:8b
# OLLAMA_TIMEOUT=120

# Logging configuration (optional)
# LOG_LEVEL=INFO
# LOG_ENABLE_FILE=true
# LOG_ENABLE_CONSOLE=true
# LOG_DETALLED_FORMAT=true
```

#### Frontend Environment Variables

The frontend connects to the backend API at `http://localhost:8000` by default. If you need to change this, create a `.env.local` file in the `frontend/` directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🗄️ Database Setup

### Initialize Database

The database is automatically initialized on first startup. To manually run migrations:

```bash
cd backend

# Activate virtual environment if not already active
source venv/bin/activate

# Run migrations
alembic upgrade head
```

### Seed Database (Optional)

To populate the database with sample routes and data:

```bash
cd backend
source venv/bin/activate
python scripts/seed_db.py
```

## ▶️ Running the Application

### Development Mode

#### Option 1: Using the Development Script

```bash
# From project root
./scripts/dev.sh
```

This starts the backend server on `http://localhost:8000`.

#### Option 2: Manual Start

**Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:3000`.

### Production Mode

**Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run build
npm start
```

## 🔧 Configuration

### Ollama Setup

1. **Install Ollama**: Follow instructions at https://ollama.ai
2. **Pull the model**:
   ```bash
   ollama pull llama3.1:8b
   ```
3. **Verify Ollama is running**:
   ```bash
   curl http://localhost:11434/api/tags
   ```
4. **Test connection** (optional):
   ```bash
   cd backend
   source venv/bin/activate
   python scripts/diagnose_ollama.py
   ```

### Database Configuration

The application uses SQLite by default. The database file is created at `backend/data/app.db` automatically.

To use a different database location, set the `DATABASE_URL` environment variable in `backend/.env`:

```env
DATABASE_URL=sqlite+aiosqlite:///path/to/your/database.db
```

For PostgreSQL (advanced), modify the connection string accordingly.

## 📚 API Documentation

Once the backend is running, API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Dashboard**: http://localhost:8000/

### Main API Endpoints

- `GET /api/profiles` - List user profiles
- `POST /api/profiles` - Create user profile
- `GET /api/routes` - List hiking routes
- `GET /api/routes/{route_id}` - Get route details
- `GET /api/souvenirs` - List user souvenirs
- `GET /api/achievements` - List achievements
- `POST /api/logs` - Log route completion

## 🛠️ Development

### Running Database Migrations

```bash
cd backend
source venv/bin/activate

# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Utility Scripts

The `backend/scripts/` directory contains utility scripts:

- `seed_db.py` - Seed database with sample data
- `import_outdooractive_routes.py` - Import routes from OutdoorActive
- `generate_all_breakpoints.py` - Generate breakpoints for routes
- `calculate_route_xp.py` - Calculate XP for routes
- `diagnose_ollama.py` - Test Ollama connection
- `delete_all_users.py` - Clear all user data (development only)

### Code Style

- **Backend**: Follow PEP 8 Python style guide
- **Frontend**: ESLint configuration included

## 🧪 Testing

### Backend Testing

```bash
cd backend
source venv/bin/activate
pytest  # If tests are configured
```

### Frontend Testing

```bash
cd frontend
npm run lint
```

## 📦 Dependencies

### Backend Dependencies

Key dependencies (see `backend/requirements.txt` for full list):
- `fastapi>=0.115.0` - Web framework
- `sqlalchemy>=2.0.0` - ORM
- `alembic>=1.13.0` - Database migrations
- `httpx>=0.27.0` - HTTP client for Ollama
- `pydantic>=2.9.0` - Data validation

### Frontend Dependencies

Key dependencies (see `frontend/package.json` for full list):
- `next@16.0.3` - React framework
- `react@19.2.0` - UI library
- `typescript@^5` - Type safety
- `tailwindcss@^4.1.9` - Styling
- `@radix-ui/*` - UI component library

## 🐛 Troubleshooting

### Backend Issues

**Database permission errors:**
```bash
cd backend
./fix_db_permissions.sh
```

**Ollama connection issues:**
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check model is installed: `ollama list`
- Run diagnostics: `python scripts/diagnose_ollama.py`

**Port already in use:**
```bash
# Kill process on port 8000
lsof -ti :8000 | xargs kill -9
```

### Frontend Issues

**Build errors:**
```bash
cd frontend
rm -rf .next node_modules
npm install
npm run dev
```

**API connection errors:**
- Verify backend is running on `http://localhost:8000`
- Check CORS settings in `backend/app/main.py`
- Verify `NEXT_PUBLIC_API_URL` in frontend `.env.local`

## 📝 License

[Specify your license here]

## 👥 Contributors

[Add contributor information]

## 🙏 Acknowledgments

- OutdoorActive for route data
- Ollama for local LLM capabilities
- FastAPI and Next.js communities

---

## 📖 Additional Resources

- **Backend API Docs**: http://localhost:8000/docs
- **Ollama Documentation**: https://github.com/ollama/ollama
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **Next.js Documentation**: https://nextjs.org/docs

For questions or issues, please open an issue in the repository.
