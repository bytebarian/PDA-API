"""create_app_settings_table

Revision ID: d89ec8d9a902
Revises: c5a2f0423a10
Create Date: 2026-05-11 20:50:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d89ec8d9a902"
down_revision: Union[str, Sequence[str], None] = "c5a2f0423a10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UUID_TYPE = sa.Uuid(as_uuid=True)
JSON_TYPE = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)


def upgrade() -> None:
    """Create the app_settings table."""
    op.create_table(
        "app_settings",
        sa.Column("id", UUID_TYPE, primary_key=True, nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False, server_default="./storage"),
        sa.Column(
            "max_file_size_bytes",
            sa.Integer(),
            nullable=False,
            server_default=sa.text(str(10 * 1024 * 1024)),
        ),
        sa.Column(
            "allowed_file_types_jsonb",
            JSON_TYPE,
            nullable=False,
            server_default=sa.text(
                "'[\"application/pdf\",\"text/plain\",\"image/png\",\"image/jpeg\"]'"
            ),
        ),
        sa.Column("ocr_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("ocr_provider", sa.String(), nullable=True, server_default="tesseract"),
        sa.Column("ocr_language", sa.String(), nullable=True, server_default="eng"),
        sa.Column("ocr_dpi", sa.Integer(), nullable=True, server_default=sa.text("300")),
        sa.Column("chunk_size", sa.Integer(), nullable=False, server_default=sa.text("1000")),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False, server_default=sa.text("200")),
        sa.Column("embedding_provider", sa.String(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column(
            "embedding_dimensions",
            sa.Integer(),
            nullable=True,
            server_default=sa.text("1536"),
        ),
        sa.Column("llm_provider", sa.String(), nullable=True, server_default="local"),
        sa.Column(
            "llm_model",
            sa.String(),
            nullable=True,
            server_default="llama3.1:8b-instruct",
        ),
        sa.Column("privacy_local_only", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("telemetry_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "extra_settings_jsonb",
            JSON_TYPE,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
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
    )


def downgrade() -> None:
    """Drop the app_settings table."""
    op.drop_table("app_settings")
