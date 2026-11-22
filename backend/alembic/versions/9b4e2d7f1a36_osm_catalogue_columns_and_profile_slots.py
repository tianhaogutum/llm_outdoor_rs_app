"""OSM catalogue columns, route geometries, places, profile slots

- routes: start coordinates, loop flag, descent, locality/district/region, source and source URL,
  activity, compact display facts (``facts_json``). ``elevation`` keeps meaning total ascent (m).
- breakpoints: ``at_m`` (position along the route) and an index on ``route_id``.
- route_geometries: simplified line + elevation profile, one row per route.
- places: catalogue localities for home-location lookup.
- demo_profiles: home name/coordinates, travel radius, distance cap, free-text intent.

All new columns are nullable, so existing rows and the legacy seed are unaffected.

Revision ID: 9b4e2d7f1a36
Revises: c1a7c85ec9fa
Create Date: 2026-11-22 23:30:00

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b4e2d7f1a36"
down_revision: Union[str, None] = "c1a7c85ec9fa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ROUTE_COLUMNS = (
    ("source", sa.String(length=32)),
    ("source_url", sa.String(length=255)),
    ("activity", sa.String(length=32)),
    ("start_lat", sa.Float()),
    ("start_lon", sa.Float()),
    ("is_loop", sa.Boolean()),
    ("descent_m", sa.Integer()),
    ("locality", sa.String(length=255)),
    ("admin_district", sa.String(length=255)),
    ("admin_region", sa.String(length=255)),
    ("facts_json", sa.Text()),
)

PROFILE_COLUMNS = (
    ("home_name", sa.String(length=255)),
    ("home_lat", sa.Float()),
    ("home_lon", sa.Float()),
    ("travel_radius_km", sa.Float()),
    ("distance_cap_km", sa.Float()),
    ("intent_text", sa.Text()),
)


def upgrade() -> None:
    with op.batch_alter_table("routes") as batch_op:
        for name, type_ in ROUTE_COLUMNS:
            batch_op.add_column(sa.Column(name, type_, nullable=True))

    with op.batch_alter_table("breakpoints") as batch_op:
        batch_op.add_column(sa.Column("at_m", sa.Integer(), nullable=True))
        batch_op.create_index("ix_breakpoints_route_id", ["route_id"], unique=False)

    with op.batch_alter_table("demo_profiles") as batch_op:
        for name, type_ in PROFILE_COLUMNS:
            batch_op.add_column(sa.Column(name, type_, nullable=True))

    op.create_table(
        "route_geometries",
        sa.Column("route_id", sa.Integer(), nullable=False),
        sa.Column("coords_json", sa.Text(), nullable=False),
        sa.Column("ele_profile_json", sa.Text(), nullable=True),
        sa.Column("ele_step_m", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["route_id"], ["routes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("route_id"),
    )

    op.create_table(
        "places",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("name_key", sa.String(length=255), nullable=False),
        sa.Column("admin_district", sa.String(length=255), nullable=True),
        sa.Column("admin_region", sa.String(length=255), nullable=True),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("accuracy_m", sa.Integer(), nullable=True),
        sa.Column("route_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_places_name_key", "places", ["name_key"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_places_name_key", table_name="places")
    op.drop_table("places")
    op.drop_table("route_geometries")

    with op.batch_alter_table("demo_profiles") as batch_op:
        for name, _ in reversed(PROFILE_COLUMNS):
            batch_op.drop_column(name)

    with op.batch_alter_table("breakpoints") as batch_op:
        batch_op.drop_index("ix_breakpoints_route_id")
        batch_op.drop_column("at_m")

    with op.batch_alter_table("routes") as batch_op:
        for name, _ in reversed(ROUTE_COLUMNS):
            batch_op.drop_column(name)
