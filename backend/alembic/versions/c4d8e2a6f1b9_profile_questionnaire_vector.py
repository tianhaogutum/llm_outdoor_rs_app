"""Profile: the questionnaire vector as created (never rewritten by feedback)

``demo_profiles.questionnaire_vector_json`` keeps the user vector the questionnaire produced. The
baseline's feedback endpoint rewrites ``user_vector_json`` after every feedback (DECISIONS.md D7.1), so
the recommender pipeline reads this column instead (docs/SERVING.md, "Original vector").

Backfill for existing rows, best effort:

- no feedback rows: the vector was never rewritten by feedback. Take the chat's pre-chat vector
  (``chat_state_json.base_vector``) if a chat happened, else ``user_vector_json`` as stored.
- with feedback rows: the stored vector is compounded. Difficulty range and distance band are rebuilt
  from ``fitness_level`` with the questionnaire's table (copied below; app code is not imported in a
  migration). Preferred tags are kept as stored: tags removed by a "not-interested" feedback cannot be
  recovered. The row gets ``"_reconstructed": true`` so this stays visible.
- no vector at all: stays NULL (the service then uses ``user_vector_json`` or an empty vector).

Revision ID: c4d8e2a6f1b9
Revises: a7c3e9d2b5f4
Create Date: 2026-11-23 12:00:00

"""
import json
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "c4d8e2a6f1b9"
down_revision: Union[str, None] = "a7c3e9d2b5f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Copy of app.services.user_profile_service.FITNESS_MAPPING at this revision.
FITNESS = {
    "beginner": {"difficulty_range": [0, 1], "max_distance_km": 8.0, "min_distance_km": 0.0},
    "intermediate": {"difficulty_range": [1, 2], "max_distance_km": 20.0, "min_distance_km": 5.0},
    "advanced": {"difficulty_range": [2, 3], "max_distance_km": 50.0, "min_distance_km": 10.0},
}


def _loads(text):
    if not text:
        return None
    try:
        v = json.loads(text)
    except (TypeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def backfill_vector(user_vector_json, chat_state_json, has_feedback):
    """The questionnaire vector for one existing row (None if there is nothing to start from)."""
    state = _loads(chat_state_json) or {}
    base = state.get("base_vector") if isinstance(state.get("base_vector"), dict) else None
    stored = _loads(user_vector_json)
    vec = base or stored
    if vec is None:
        return None
    if not has_feedback:
        return dict(vec)
    out = dict(vec)
    fit = FITNESS.get(out.get("fitness_level"))
    if fit:
        out.update({k: (list(v) if isinstance(v, list) else v) for k, v in fit.items()})
    out["_reconstructed"] = True
    return out


def upgrade() -> None:
    with op.batch_alter_table("demo_profiles") as batch_op:
        batch_op.add_column(sa.Column("questionnaire_vector_json", sa.Text(), nullable=True))

    conn = op.get_bind()
    with_feedback = {r[0] for r in conn.execute(sa.text("SELECT DISTINCT demo_profile_id FROM profile_feedback"))}
    rows = conn.execute(sa.text("SELECT id, user_vector_json, chat_state_json FROM demo_profiles")).fetchall()
    for pid, uv, cs in rows:
        vec = backfill_vector(uv, cs, pid in with_feedback)
        if vec is not None:
            conn.execute(
                sa.text("UPDATE demo_profiles SET questionnaire_vector_json = :v WHERE id = :id"),
                {"v": json.dumps(vec, ensure_ascii=False), "id": pid},
            )


def downgrade() -> None:
    with op.batch_alter_table("demo_profiles") as batch_op:
        batch_op.drop_column("questionnaire_vector_json")
