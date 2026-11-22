"""
Script to seed or update achievements in the database.
Run this to ensure all achievements from achievement_service.py are in the database.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db_session, init_db
from app.services.achievement_service import seed_achievements
from app.settings import get_settings


async def main() -> None:
    """Seed achievements in the database."""
    print("🏆 Seeding achievements...")

    settings = get_settings()
    init_db(settings)

    session = await get_db_session()
    try:
        await seed_achievements(session)
        print("✅ Achievements seeded successfully!")
        print("   Check your database to see the new achievements.")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
