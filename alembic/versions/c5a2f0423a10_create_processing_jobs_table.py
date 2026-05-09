"""create_processing_jobs_table

Revision ID: c5a2f0423a10
Revises: 93f56f34f1ae
Create Date: 2026-05-09 22:05:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c5a2f0423a10"
down_revision: Union[str, Sequence[str], None] = "93f56f34f1ae"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UUID_TYPE = sa.Uuid(as_uuid=True)
JSON_TYPE = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)


def upgrade() -> None:
    """Create the processing_jobs table."""
    op.create_table(
        "processing_jobs",
        sa.Column("id", UUID_TYPE, primary_key=True, nullable=False),
        sa.Column("document_id", UUID_TYPE, nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="awaiting"),
        sa.Column("stage", sa.String(), nullable=False, server_default="queued"),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("3"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("error_details_jsonb", JSON_TYPE, nullable=True),
        sa.Column("stage_history_jsonb", JSON_TYPE, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_processing_jobs_document_id",
        "processing_jobs",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "ix_processing_jobs_document_id_status",
        "processing_jobs",
        ["document_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the processing_jobs table."""
    op.drop_index("ix_processing_jobs_document_id_status", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_document_id", table_name="processing_jobs")
    op.drop_table("processing_jobs")
