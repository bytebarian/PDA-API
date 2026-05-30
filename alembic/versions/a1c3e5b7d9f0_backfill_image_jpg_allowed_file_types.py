"""backfill_image_jpg_allowed_file_types

Revision ID: a1c3e5b7d9f0
Revises: d89ec8d9a902
Create Date: 2026-05-30 20:37:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1c3e5b7d9f0"
down_revision: Union[str, Sequence[str], None] = "d89ec8d9a902"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSON_TYPE = sa.JSON().with_variant(
    postgresql.JSONB(astext_type=sa.Text()),
    "postgresql",
)

_NEW_DEFAULT = (
    '["application/pdf","text/plain","image/png","image/jpeg","image/jpg"]'
)
_OLD_DEFAULT = '["application/pdf","text/plain","image/png","image/jpeg"]'


def upgrade() -> None:
    """Update server default and backfill existing rows to include image/jpg."""
    # Update the column server default so future rows seeded at DB level get
    # the full list including image/jpg.  batch_alter_table is used so that
    # SQLite (which cannot ALTER COLUMN in-place) recreates the table correctly.
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.alter_column(
            "allowed_file_types_jsonb",
            server_default=sa.text(f"'{_NEW_DEFAULT}'"),
            existing_type=JSON_TYPE,
            existing_nullable=False,
        )

    # Backfill existing rows whose allowed_file_types_jsonb still matches the
    # old default (i.e. does not yet contain "image/jpg").
    connection = op.get_bind()
    dialect = connection.dialect.name

    if dialect == "postgresql":
        # PostgreSQL: use JSONB containment to find rows missing "image/jpg"
        # and append it with the array concatenation operator.
        connection.execute(
            sa.text(
                "UPDATE app_settings "
                "SET allowed_file_types_jsonb = allowed_file_types_jsonb || '[\"image/jpg\"]'::jsonb "
                "WHERE NOT (allowed_file_types_jsonb @> '\"image/jpg\"'::jsonb)"
            )
        )
    else:
        # SQLite / generic path: fetch all rows, patch in Python, write back.
        rows = connection.execute(
            sa.select(
                sa.column("id", sa.String()),
                sa.column("allowed_file_types_jsonb", sa.JSON()),
            ).select_from(sa.table("app_settings"))
        ).fetchall()

        import json

        for row in rows:
            row_id, file_types = row[0], row[1]
            if isinstance(file_types, str):
                file_types = json.loads(file_types)
            if "image/jpg" not in file_types:
                updated = file_types + ["image/jpg"]
                connection.execute(
                    sa.text(
                        "UPDATE app_settings "
                        "SET allowed_file_types_jsonb = :val "
                        "WHERE id = :id"
                    ),
                    {"val": json.dumps(updated), "id": str(row_id)},
                )


def downgrade() -> None:
    """Restore the previous server default; remove image/jpg from existing rows."""
    # Remove "image/jpg" from existing rows first, before narrowing the default.
    connection = op.get_bind()
    dialect = connection.dialect.name

    if dialect == "postgresql":
        connection.execute(
            sa.text(
                "UPDATE app_settings "
                "SET allowed_file_types_jsonb = ("
                "  SELECT jsonb_agg(elem) "
                "  FROM jsonb_array_elements(allowed_file_types_jsonb) AS elem "
                "  WHERE elem::text <> '\"image/jpg\"'"
                ") "
                "WHERE allowed_file_types_jsonb @> '\"image/jpg\"'::jsonb"
            )
        )
    else:
        import json

        rows = connection.execute(
            sa.select(
                sa.column("id", sa.String()),
                sa.column("allowed_file_types_jsonb", sa.JSON()),
            ).select_from(sa.table("app_settings"))
        ).fetchall()

        for row in rows:
            row_id, file_types = row[0], row[1]
            if isinstance(file_types, str):
                file_types = json.loads(file_types)
            if "image/jpg" in file_types:
                updated = [ft for ft in file_types if ft != "image/jpg"]
                connection.execute(
                    sa.text(
                        "UPDATE app_settings "
                        "SET allowed_file_types_jsonb = :val "
                        "WHERE id = :id"
                    ),
                    {"val": json.dumps(updated), "id": str(row_id)},
                )

    # Restore the old server default.
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.alter_column(
            "allowed_file_types_jsonb",
            server_default=sa.text(f"'{_OLD_DEFAULT}'"),
            existing_type=JSON_TYPE,
            existing_nullable=False,
        )
