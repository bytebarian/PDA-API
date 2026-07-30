"""backfill_llm_model_to_canonical_default

Revision ID: e1b2c3d4f5a6
Revises: c7e3b1a4f9d2
Create Date: 2026-07-30 20:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1b2c3d4f5a6"
down_revision: Union[str, Sequence[str], None] = "c7e3b1a4f9d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CANONICAL_DEFAULT = "llama3.1:8b-instruct-q8_0"
_SUPPORTED_MODELS = frozenset(
    {
        "llama3.1:8b-instruct-q8_0",
        "llama3.1:8b",
        "llama3.2:3b",
        "gemma3:1b",
    }
)

# Models that should be replaced with the canonical default because they are
# obsolete identifiers (no longer valid Ollama tags in this application).
_OBSOLETE_MODELS = ("llama3.1:8b-instruct",)


def upgrade() -> None:
    """Fix server default and backfill obsolete llm_model values."""
    app_settings = sa.table(
        "app_settings",
        sa.column("llm_model", sa.String),
    )

    # Replace each obsolete model value with the canonical default, but only
    # when the current value is NOT already a supported model identifier.
    for obsolete in _OBSOLETE_MODELS:
        op.execute(
            app_settings.update()
            .where(app_settings.c.llm_model == obsolete)
            .values(llm_model=_CANONICAL_DEFAULT)
        )

    # Also replace NULL values with the canonical default.
    op.execute(
        app_settings.update()
        .where(app_settings.c.llm_model.is_(None))
        .values(llm_model=_CANONICAL_DEFAULT)
    )

    # Update the server-side default so that fresh rows use the canonical value.
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.alter_column(
            "llm_model",
            server_default=_CANONICAL_DEFAULT,
            existing_type=sa.String(),
            existing_nullable=True,
        )


def downgrade() -> None:
    """Restore the original server default; cannot reverse row-level backfill."""
    with op.batch_alter_table("app_settings") as batch_op:
        batch_op.alter_column(
            "llm_model",
            server_default="llama3.1:8b-instruct",
            existing_type=sa.String(),
            existing_nullable=True,
        )
