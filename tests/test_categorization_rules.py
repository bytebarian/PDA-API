"""Unit tests for the rule-based categorization provider.

All tests are deterministic and fixture-based.  No external services are
required.
"""

from __future__ import annotations

import pytest

from app.adapters.categorization.base import DocumentCategory
from app.adapters.categorization.rules import RulesCategorizationProvider


@pytest.fixture
def provider() -> RulesCategorizationProvider:
    return RulesCategorizationProvider()


# ---------------------------------------------------------------------------
# Contract / agreement detection
# ---------------------------------------------------------------------------


async def test_rules_classifies_contract_text(provider: RulesCategorizationProvider) -> None:
    """Obvious contract text must be classified as 'contract'."""
    text = (
        "This contract is entered into by and between the parties. "
        "The notice period is 30 days. All obligations and clauses herein are binding. "
        "Termination requires written consent."
    )
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.contract.value
    assert result.confidence > 0.0
    assert result.source == "rules"


async def test_rules_classifies_agreement_text(provider: RulesCategorizationProvider) -> None:
    """NDA / confidentiality agreement text must be classified as 'agreement'."""
    text = (
        "This agreement is signed by both parties. "
        "Non-disclosure and confidentiality obligations apply. "
        "This is a non-compete agreement with binding terms."
    )
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.agreement.value
    assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Offer / quotation detection
# ---------------------------------------------------------------------------


async def test_rules_classifies_offer_text(provider: RulesCategorizationProvider) -> None:
    """Price offer / quotation text must be classified as 'offer'."""
    text = (
        "We are pleased to present this proposal and quotation. "
        "The price offer is valid until December 31, 2026. "
        "Please confirm acceptance within 14 days."
    )
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.offer.value
    assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Invoice detection
# ---------------------------------------------------------------------------


async def test_rules_classifies_invoice_text(provider: RulesCategorizationProvider) -> None:
    """Invoice text must be classified as 'invoice'."""
    text = (
        "INVOICE #12345\n"
        "VAT number: DE123456789\n"
        "Total amount: €2,500.00\n"
        "Payment due: 2026-07-01\n"
        "Bank account: DE89 3704 0044 0532 0130 00"
    )
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.invoice.value
    assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Official document detection
# ---------------------------------------------------------------------------


async def test_rules_classifies_official_document_text(provider: RulesCategorizationProvider) -> None:
    """Identity card / passport text must be classified as 'official_document'."""
    text = (
        "IDENTITY CARD\n"
        "National ID: PL-1234567\n"
        "Certificate of residence issued by the authority."
    )
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.official_document.value
    assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Documentation detection
# ---------------------------------------------------------------------------


async def test_rules_classifies_documentation_text(provider: RulesCategorizationProvider) -> None:
    """Technical manual / specification must be classified as 'documentation'."""
    text = (
        "Technical documentation for the PDA API.\n"
        "This manual provides the specification and requirements.\n"
        "See the procedure section for installation guide."
    )
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.documentation.value
    assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Note / memo detection
# ---------------------------------------------------------------------------


async def test_rules_classifies_note_text(provider: RulesCategorizationProvider) -> None:
    """Meeting notes / personal memo must be classified as 'note'."""
    text = (
        "Meeting notes – 2026-06-02\n"
        "Memo: reminder to submit the draft before Friday.\n"
        "Personal note: review the minutes."
    )
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.note.value
    assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# Low-signal / fallback
# ---------------------------------------------------------------------------


async def test_rules_returns_other_for_low_signal_text(
    provider: RulesCategorizationProvider,
) -> None:
    """Text with no recognizable keywords must fall back to 'other'."""
    text = "The quick brown fox jumps over the lazy dog."
    result = await provider.categorize(text)
    assert result.category == DocumentCategory.other.value
    assert result.confidence == 0.0


async def test_rules_returns_other_for_empty_text(
    provider: RulesCategorizationProvider,
) -> None:
    """Empty text must return 'other' deterministically."""
    result = await provider.categorize("")
    assert result.category == DocumentCategory.other.value
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


async def test_rules_is_deterministic(provider: RulesCategorizationProvider) -> None:
    """Same input must always produce the same output."""
    text = "Invoice #999 VAT number total amount payment due bank account."
    r1 = await provider.categorize(text)
    r2 = await provider.categorize(text)
    assert r1.category == r2.category
    assert r1.confidence == r2.confidence
    assert r1.reason == r2.reason


# ---------------------------------------------------------------------------
# Summary signal
# ---------------------------------------------------------------------------


async def test_rules_uses_summary_as_signal(provider: RulesCategorizationProvider) -> None:
    """A strong summary signal should influence category assignment."""
    # Low-signal body text but strong summary
    result = await provider.categorize(
        "Some generic document content.",
        summary="This is an invoice for services rendered. VAT and total amount included.",
    )
    assert result.category == DocumentCategory.invoice.value


# ---------------------------------------------------------------------------
# Filename signal
# ---------------------------------------------------------------------------


async def test_rules_uses_filename_as_signal(provider: RulesCategorizationProvider) -> None:
    """Filename containing obvious category hint should contribute to scoring."""
    result = await provider.categorize(
        "Some generic document content.",
        filename="invoice_2026_june.pdf",
    )
    assert result.category == DocumentCategory.invoice.value


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


async def test_rules_healthcheck_returns_true(provider: RulesCategorizationProvider) -> None:
    """Rules provider healthcheck must always return True."""
    assert await provider.healthcheck() is True
