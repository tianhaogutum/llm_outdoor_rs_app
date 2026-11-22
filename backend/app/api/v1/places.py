"""
Place lookup for the home-location slot: typed name -> catalogue locality with coordinates.

GET /api/places?q=garmi&limit=8 ranks exact, prefix, word-prefix, then fuzzy matches
(see app/services/place_service.py). No external geocoder is called.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import PlaceResponse, PlaceSearchResponse
from app.database import get_db
from app.services.place_service import search_places

router = APIRouter(prefix="/places", tags=["places"])


@router.get("", response_model=PlaceSearchResponse)
async def find_places(
    db: Annotated[AsyncSession, Depends(get_db)],
    q: Annotated[str, Query(min_length=1, max_length=100, description="typed place name")],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> PlaceSearchResponse:
    matches = await search_places(db, q, limit)
    places = [
        PlaceResponse(
            id=m.place.id,
            name=m.place.name,
            admin_district=m.place.admin_district,
            admin_region=m.place.admin_region,
            lat=m.place.lat,
            lon=m.place.lon,
            accuracy_m=m.place.accuracy_m,
            route_count=m.place.route_count,
            match=m.match,
            score=m.score,
        )
        for m in matches
    ]
    return PlaceSearchResponse(places=places, total=len(places))
