"""extend_document_chunks_for_vector_search

Revision ID: e7a9b2f4d3c1
Revises: a1c3e5b7d9f0
Create Date: 2026-06-01 19:50:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7a9b2f4d3c1"
down_revision: Union[str, Sequence[str], None] = "a1c3e5b7d9f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "document_chunks",
        sa.Column("embedding_provider", sa.String(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("embedding_created_at", sa.DateTime(timezone=True), nullable=True),
    )

    if is_postgres:
        op.execute(
            "UPDATE document_chunks SET embedding_dimension = vector_dims(embedding) "
            "WHERE embedding IS NOT NULL AND embedding_dimension IS NULL"
        )
    else:
        op.execute(
            "UPDATE document_chunks SET embedding_dimension = json_array_length(embedding) "
            "WHERE embedding IS NOT NULL AND embedding_dimension IS NULL"
        )

    op.create_index(
        "ix_document_chunks_embedding_model",
        "document_chunks",
        ["embedding_model"],
        unique=False,
    )

    op.execute("UPDATE document_chunks SET metadata_jsonb = '{}' WHERE metadata_jsonb IS NULL")
    if is_postgres:
        op.alter_column(
            "document_chunks",
            "metadata_jsonb",
            server_default=sa.text("'{}'"),
            nullable=False,
        )

    if is_postgres:
        op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector")
        # A dimensionless vector column can store embeddings produced by models
        # with different dimensions, but pgvector cannot build one ANN index
        # across those rows. Such indexes must be expression/partial indexes
        # scoped to a specific model and dimension.


def downgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    if is_postgres:
        op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
        op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_ivfflat")
        op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(1536)")

    if is_postgres:
        op.alter_column(
            "document_chunks",
            "metadata_jsonb",
            server_default=None,
            nullable=True,
        )
    op.drop_index("ix_document_chunks_embedding_model", table_name="document_chunks")
    op.drop_column("document_chunks", "embedding_created_at")
    op.drop_column("document_chunks", "embedding_dimension")
    op.drop_column("document_chunks", "embedding_provider")
