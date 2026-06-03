"""add_full_text_search_gin_index_on_chunks

Revision ID: c7e3b1a4f9d2
Revises: b6c8d1e4f2a7
Create Date: 2026-06-03 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7e3b1a4f9d2"
down_revision: Union[str, Sequence[str], None] = "b6c8d1e4f2a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_document_chunks_content_tsvector "
            "ON document_chunks "
            "USING GIN (to_tsvector('simple', content))"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "DROP INDEX IF EXISTS ix_document_chunks_content_tsvector"
        )
