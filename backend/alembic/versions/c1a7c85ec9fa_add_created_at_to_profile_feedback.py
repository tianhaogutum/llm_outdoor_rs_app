"""add created_at to profile_feedback

Feedback needs a timestamp so the recommender's time decay can use its real age.
Existing rows get the migration time (CURRENT_TIMESTAMP, UTC).

Revision ID: c1a7c85ec9fa
Revises: e4fda692220c
Create Date: 2026-11-22 22:00:00

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1a7c85ec9fa"
down_revision: Union[str, None] = "e4fda692220c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite cannot ALTER a column in place; batch mode recreates the table. A NOT NULL column with a
    # server default is filled for existing rows during the copy.
    with op.batch_alter_table("profile_feedback") as batch_op:
        batch_op.add_column(
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("profile_feedback") as batch_op:
        batch_op.drop_column("created_at")
