"""add_summary_metadata_to_documents

Revision ID: f3c1a9e2b4d7
Revises: e7a9b2f4d3c1
Create Date: 2026-06-02 08:00:00.000000

Adds summarization traceability columns to the documents table:

  summary_model          – which local model produced the summary
  summary_generated_at   – when the summary was generated
  summary_status         – lifecycle status (pending/processing/ready/failed/skipped)
  summary_error          – concise error reason when summary_status is failed/skipped
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3c1a9e2b4d7"
down_revision: Union[str, Sequence[str], None] = "e7a9b2f4d3c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add summary metadata columns to the documents table."""
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column("summary_model", sa.String(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "summary_generated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "summary_status",
                sa.String(),
                nullable=False,
                server_default="pending",
            )
        )
        batch_op.add_column(
            sa.Column("summary_error", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    """Remove summary metadata columns from the documents table."""
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("summary_error")
        batch_op.drop_column("summary_status")
        batch_op.drop_column("summary_generated_at")
        batch_op.drop_column("summary_model")
