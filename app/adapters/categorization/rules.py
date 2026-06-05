"""Deterministic rule-based categorization provider.

The classifier scores each category by counting keyword hits across normalized
text, summary, and filename signals. Ties are resolved by category order
(higher priority categories come first).

The returned confidence score can be used by the calling service to demote
low-confidence results to ``other``.

Keyword sets are intentionally compact and cover English.  Adding keywords for
additional languages is straightforward – just extend the lists below.
"""

from __future__ import annotations

import re

from app.adapters.categorization.base import (
    CategorizationResult,
    CategorizationSource,
    DocumentCategory,
)

# ---------------------------------------------------------------------------
# Keyword definitions
# ---------------------------------------------------------------------------
# Each entry is (category, keywords_list).  The order matters for tie-breaking:
# more specific categories appear first so that a document that matches both
# "contract" and "agreement" keywords gets "contract" when scores are equal.

_RULES: list[tuple[str, list[str]]] = [
    (
        DocumentCategory.invoice.value,
        [
            "invoice",
            "vat",
            "tax number",
            "total amount",
            "payment due",
            "bank account",
            "invoice number",
            "billing",
            "amount due",
            "subtotal",
            "faktura",           # Polish
            "rechnung",          # German
            "facture",           # French
        ],
    ),
    (
        DocumentCategory.contract.value,
        [
            "contract",
            "termination",
            "notice period",
            "obligation",
            "clause",
            "hereby agrees",
            "binding agreement",
            "whereas",
            "in witness whereof",
            "kontrakt",          # Polish/German
            "vertrag",           # German
        ],
    ),
    (
        DocumentCategory.agreement.value,
        [
            "agreement",
            "terms",
            "parties",
            "signed",
            "non-compete",
            "confidentiality",
            "nda",
            "non disclosure",
            "mutual agreement",
            "umowa",             # Polish
            "vereinbarung",      # German
        ],
    ),
    (
        DocumentCategory.offer.value,
        [
            "offer",
            "proposal",
            "quotation",
            "price offer",
            "valid until",
            "acceptance",
            "quote",
            "bid",
            "tender",
            "oferta",            # Polish/Spanish
            "angebot",           # German
        ],
    ),
    (
        DocumentCategory.official_document.value,
        [
            "passport",
            "identity card",
            "national id",
            "certificate",
            "decision",
            "permit",
            "authority",
            "office",
            "government",
            "official",
            "license",
            "registration",
            "dowód osobisty",    # Polish
            "ausweis",           # German
        ],
    ),
    (
        DocumentCategory.documentation.value,
        [
            "manual",
            "specification",
            "documentation",
            "requirements",
            "technical description",
            "procedure",
            "guide",
            "handbook",
            "readme",
            "api reference",
            "instrukcja",        # Polish
            "handbuch",          # German
        ],
    ),
    (
        DocumentCategory.note.value,
        [
            "note",
            "memo",
            "meeting notes",
            "draft",
            "reminder",
            "personal note",
            "todo",
            "minutes",
            "notatka",           # Polish
            "notiz",             # German
        ],
    ),
]

# NOTE: Confidence is computed per-category based on matched keyword fraction.


def _tokenize(text: str) -> str:
    """Lower-case the text for case-insensitive keyword matching."""
    return text.lower()


def _count_hits(haystack: str, keywords: list[str]) -> int:
    """Count how many distinct keywords appear in *haystack*."""
    count = 0
    for kw in keywords:
        # Match whole keywords/phrases (avoid partial word matches like "note" in "noted").
        pattern = rf"(?<!\w){re.escape(kw)}(?!\w)"
        if re.search(pattern, haystack):
            count += 1
    return count


class RulesCategorizationProvider:
    """Classify documents using a deterministic keyword-scoring approach."""

    name: str = CategorizationSource.rules.value

    async def categorize(
        self,
        text: str,
        *,
        filename: str | None = None,
        summary: str | None = None,
        metadata: dict | None = None,
    ) -> CategorizationResult:
        """Assign a category based on keyword density in document signals."""
        haystack_parts: list[str] = []
        if text:
            haystack_parts.append(text)
        if summary:
            # Give the summary a small boost by repeating it.
            haystack_parts.append(summary)
            haystack_parts.append(summary)
        if filename:
            # Treat separators as token boundaries so "invoice_2026.pdf" matches "invoice".
            normalized_filename = re.sub(r"[_\-.]+", " ", filename)
            haystack_parts.append(normalized_filename)

        haystack = _tokenize(" ".join(haystack_parts))

        best_category = DocumentCategory.other.value
        best_score = 0
        best_keywords_count = 0
        matched_keywords: list[str] = []

        for category, keywords in _RULES:
            hits = _count_hits(haystack, keywords)
            if hits > best_score:
                best_score = hits
                best_category = category
                best_keywords_count = len(keywords)
                matched_keywords = [
                    kw
                    for kw in keywords
                    if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", haystack)
                ]

        if best_score == 0:
            return CategorizationResult(
                category=DocumentCategory.other.value,
                confidence=0.0,
                reason="no_keywords_matched",
                source=CategorizationSource.rules.value,
            )

        # Confidence = fraction of that category's keywords that matched.
        confidence = round(best_score / best_keywords_count, 4)

        reason = (
            f"matched {best_score}/{best_keywords_count} keywords: "
            + ", ".join(matched_keywords[:5])
            + ("..." if len(matched_keywords) > 5 else "")
        )

        return CategorizationResult(
            category=best_category,
            confidence=confidence,
            reason=reason,
            source=CategorizationSource.rules.value,
        )

    async def healthcheck(self) -> bool:
        return True
