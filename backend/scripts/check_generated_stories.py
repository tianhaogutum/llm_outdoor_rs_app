#!/usr/bin/env python3
"""
Query routes with generated stories in database
Usage: python backend/scripts/check_generated_stories.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.database import init_db, get_db_session
from app.models.entities import Route, Breakpoint


async def check_generated_stories():
    """Query and display all routes with generated stories"""
    print("=" * 80)
    print("📖 Querying generated stories from database")
    print("=" * 80)
    
    # Initialize database
    init_db()
    
    # Get database session
    db = await get_db_session()
    
    try:
        # Query all routes with stories
        result = await db.execute(
            select(Route)
            .where(Route.story_prologue_body.isnot(None))
            .options(selectinload(Route.breakpoints))
            .order_by(Route.id)
        )
        routes_with_stories = result.scalars().all()
        
        # Statistics
        total_result = await db.execute(
            select(func.count(Route.id))
        )
        total_routes = total_result.scalar() or 0
        
        print(f"\n📊 Statistics:")
        print(f"   Total routes: {total_routes}")
        print(f"   Routes with stories: {len(routes_with_stories)}")
        print(f"   Routes without stories: {total_routes - len(routes_with_stories)}")
        
        if routes_with_stories:
            print(f"\n✅ Routes with generated stories:")
            print("-" * 80)
            for i, route in enumerate(routes_with_stories, 1):
                # Count breakpoints
                bp_count = len(route.breakpoints) if route.breakpoints else 0
                
                # Count breakpoints with main_quest_snippet
                bp_with_story = 0
                if route.breakpoints:
                    bp_with_story = sum(
                        1 for bp in route.breakpoints 
                        if bp.main_quest_snippet
                    )
                
                # Calculate story content length
                prologue_len = len(route.story_prologue_body) if route.story_prologue_body else 0
                epilogue_len = len(route.story_epilogue_body) if route.story_epilogue_body else 0
                
                print(f"\n{i}. Route ID: {route.id}")
                print(f"   Title: {route.title}")
                if route.story_prologue_title:
                    print(f"   Story Title: {route.story_prologue_title}")
                print(f"   Location: {route.location or 'N/A'}")
                print(f"   Difficulty: {route.difficulty or 'N/A'}")
                print(f"   Length: {route.length_meters/1000:.2f} km" if route.length_meters else "   Length: N/A")
                print(f"   Breakpoints: {bp_count} ({bp_with_story} have chapter content)")
                print(f"   Prologue length: {prologue_len} characters")
                print(f"   Epilogue length: {epilogue_len} characters")
                
                # Display prologue preview
                if route.story_prologue_body:
                    preview = route.story_prologue_body[:100].replace('\n', ' ')
                    print(f"   Prologue preview: {preview}...")
        else:
            print("\n❌ No generated stories found in database")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(check_generated_stories())



