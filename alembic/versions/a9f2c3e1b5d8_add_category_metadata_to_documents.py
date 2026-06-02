"""add_category_metadata_to_documents

Revision ID: a9f2c3e1b5d8
Revises: f3c1a9e2b4d7
Create Date: 2026-06-02 09:00:00.000000

Adds categorization traceability columns to the documents table:

  category_source        – which provider assigned the category (rules/local_model/manual/fallback/mock)
  category_confidence    – confidence score between 0.0 and 1.0
  category_reason        – short diagnostic explanation of why the category was chosen
  category_model         – which local model was used (for local_model source)
  category_generated_at  – when the category was assigned
  category_status        – lifecycle status (pending/processing/ready/failed/skipped)
  category_error         – concise error reason when category_status is failed/skipped
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a9f2c3e1b5d8"
down_revision: Union[str, Sequence[str], None] = "f3c1a9e2b4d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add category metadata columns to the documents table."""
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column("category_source", sa.String(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("category_confidence", sa.Float(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("category_reason", sa.Text(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("category_model", sa.String(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "category_generated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "category_status",
                sa.String(),
                nullable=False,
                server_default="pending",
            )
        )
        batch_op.add_column(
            sa.Column("category_error", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    """Remove category metadata columns from the documents table."""
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("category_error")
        batch_op.drop_column("category_status")
        batch_op.drop_column("category_generated_at")
        batch_op.drop_column("category_model")
        batch_op.drop_column("category_reason")
        batch_op.drop_column("category_confidence")
        batch_op.drop_column("category_source")
