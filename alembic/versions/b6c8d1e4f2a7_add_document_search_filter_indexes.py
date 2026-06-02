"""add_document_search_filter_indexes

Revision ID: b6c8d1e4f2a7
Revises: a9f2c3e1b5d8
Create Date: 2026-06-02 21:10:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b6c8d1e4f2a7"
down_revision: Union[str, Sequence[str], None] = "a9f2c3e1b5d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_documents_status", "documents", ["status"], unique=False)
    op.create_index("ix_documents_category", "documents", ["category"], unique=False)
    op.create_index("ix_documents_file_type", "documents", ["file_type"], unique=False)
    op.create_index("ix_documents_created_at", "documents", ["created_at"], unique=False)
    op.create_index("ix_documents_updated_at", "documents", ["updated_at"], unique=False)

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_documents_metadata_jsonb_gin "
            "ON documents USING GIN (metadata_jsonb)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_documents_metadata_jsonb_gin")

    op.drop_index("ix_documents_updated_at", table_name="documents")
    op.drop_index("ix_documents_created_at", table_name="documents")
    op.drop_index("ix_documents_file_type", table_name="documents")
    op.drop_index("ix_documents_category", table_name="documents")
    op.drop_index("ix_documents_status", table_name="documents")
