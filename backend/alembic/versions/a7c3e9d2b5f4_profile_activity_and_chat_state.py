"""Profile activity slot and server-side chat state

- demo_profiles.activity: hiking | cycling | mtb | running, set by the chat (Phase 4); the UI tab and the
  recommender's category filter follow it. Null = not stated.
- demo_profiles.chat_state_json: the conversation (turns, clarifying-turn count, pending question,
  chat-only slots such as difficulty/distance bands and avoided tags, the questionnaire vector as it
  was before the first chat turn). Schema: app/chat/schema.py ``ChatState``.

Both nullable: existing profiles are unaffected.

Revision ID: a7c3e9d2b5f4
Revises: 9b4e2d7f1a36
Create Date: 2026-11-22 23:59:00

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

revision: str = "a7c3e9d2b5f4"
down_revision: Union[str, None] = "9b4e2d7f1a36"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("demo_profiles") as batch_op:
        batch_op.add_column(sa.Column("activity", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("chat_state_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("demo_profiles") as batch_op:
        batch_op.drop_column("chat_state_json")
        batch_op.drop_column("activity")
