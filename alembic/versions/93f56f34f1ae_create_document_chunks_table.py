"""create_document_chunks_table

Revision ID: 93f56f34f1ae
Revises: 1b8bb4d20971
Create Date: 2026-05-09 20:55:00.000000

"""

from typing import Sequence, Union

from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "93f56f34f1ae"
down_revision: Union[str, Sequence[str], None] = "1b8bb4d20971"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UUID_TYPE = sa.Uuid(as_uuid=True)
METADATA_TYPE = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)
EMBEDDING_TYPE = Vector(1536).with_variant(sa.JSON(), "sqlite")


def upgrade() -> None:
    """Create the document_chunks table."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "document_chunks",
        sa.Column("id", UUID_TYPE, primary_key=True, nullable=False),
        sa.Column("document_id", UUID_TYPE, nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("source_start_offset", sa.Integer(), nullable=True),
        sa.Column("source_end_offset", sa.Integer(), nullable=True),
        sa.Column("metadata_jsonb", METADATA_TYPE, nullable=True),
        sa.Column("embedding", EMBEDDING_TYPE, nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
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
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_id_chunk_index",
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "ix_document_chunks_document_id_chunk_index",
        "document_chunks",
        ["document_id", "chunk_index"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the document_chunks table."""
    op.drop_index("ix_document_chunks_document_id_chunk_index", table_name="document_chunks")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
