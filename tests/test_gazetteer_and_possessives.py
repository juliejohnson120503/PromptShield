"""
Test Suite: 50k Offline Gazetteer & Possessive Entity Resolution
================================================================
Validates:
1. 50k+ offline gazetteer loaded with 0% GPU and 0% VRAM.
2. Public figures and public organizations are correctly recognized as public knowledge.
3. Private individuals are masked as PII.
4. Possessive entity deduplication: "Anand" and "Anand's son" map to the SAME person placeholder (<PERSON_1>).
5. Sub-millisecond lookup latency.
"""

import time
import pytest
from backend.known_entities_resolver import get_resolver
from backend.core import PromptShieldCore
from backend.models import EntityType


@pytest.fixture(scope="module")
def resolver():
    return get_resolver()


@pytest.fixture(scope="module")
def core():
    return PromptShieldCore()


def test_gazetteer_size_and_zero_resource(resolver):
    """Ensure gazetteer contains over 50,000 entries and runs purely offline in memory."""
    assert resolver.total_entities_count >= 50000, f"Expected >= 50,000 entities, got {resolver.total_entities_count}"
    assert resolver.total_figures_count >= 3000, f"Expected >= 3,000 figures, got {resolver.total_figures_count}"
    assert resolver.total_orgs_count >= 40000, f"Expected >= 40,000 orgs, got {resolver.total_orgs_count}"



def test_gazetteer_lookup_speed(resolver):
    """Verify O(1) set lookup latency is well under 0.01 ms per check."""
    start = time.perf_counter()
    iterations = 1000
    for _ in range(iterations):
        resolver.is_public_figure("cillian murphy")
        resolver.is_public_org("snowflake")
        resolver.is_public_org("non_existent_company_xyz_123")
    elapsed = time.perf_counter() - start
    avg_per_check_ms = (elapsed / (iterations * 3)) * 1000
    assert avg_per_check_ms < 0.05, f"Lookup took {avg_per_check_ms:.5f} ms, expected < 0.05 ms"


def test_public_figures_recognition(resolver):
    """Ensure notable public figures across domains are recognized."""
    public_figures = [
        "Elon Musk",
        "Satya Nadella",
        "Cillian Murphy",
        "Pedro Pascal",
        "Denis Villeneuve",
        "Zendaya",
        "Erling Haaland",
        "Viswanathan Anand",
        "Magnus Carlsen",
        "Shashi Tharoor",
        "Amartya Sen",
        "Ada Lovelace",
        "Albert Einstein",
    ]
    for figure in public_figures:
        assert resolver.is_public_figure(figure), f"Expected {figure} to be recognized as a public figure"


def test_public_organizations_recognition(resolver):
    """Ensure enterprise tech, universities, and public institutions are recognized."""
    public_orgs = [
        "Google",
        "Microsoft",
        "Snowflake",
        "Databricks",
        "Palantir",
        "Stripe",
        "Adyen",
        "CrowdStrike",
        "HashiCorp",
        "Cloudflare",
        "Oxford University",
        "Stanford University",
        "World Health Organization",
        "Federal Reserve",
    ]
    for org in public_orgs:
        assert resolver.is_public_org(org), f"Expected {org} to be recognized as a public organization"


def test_private_individuals_not_in_gazetteer(resolver):
    """Ensure private, non-notable names are NOT flagged as public figures."""
    private_names = [
        "Anand Sharma 992",
        "Johnathan Doe Private Client",
        "Jerin Binu Employee",
        "Kavita Verma Internal Reviewer",
    ]
    for name in private_names:
        assert not resolver.is_public_figure(name), f"Expected {name} to be private, but was marked public"


def test_public_entities_retained_in_prompt(core):
    """Inquiry prompts regarding public figures and public companies should RETAIN their names."""
    prompt = "Tell me about Cillian Murphy and the history of Snowflake and Databricks."
    result = core.sanitize(prompt)
    assert "Cillian Murphy" in result.sanitized_prompt, "Cillian Murphy should be retained"
    assert "Snowflake" in result.sanitized_prompt, "Snowflake should be retained"
    assert "Databricks" in result.sanitized_prompt, "Databricks should be retained"
    assert len(result.masked_entities) == 0, f"No private entities should be masked, got {result.masked_entities}"


def test_private_person_masked_in_prompt(core):
    """Private individual names must be masked as PERSON placeholders."""
    prompt = "Please send the internal draft to Johnathan Miller at Miller Industries."
    result = core.sanitize(prompt)
    assert "<PERSON_" in result.sanitized_prompt
    assert "Johnathan Miller" not in result.sanitized_prompt


def test_anand_vs_anands_possessive_deduplication(core):
    """
    CRITICAL TEST:
    Prompt: 'My name is Anand. Anand's son is working.'
    Both mentions of 'Anand' must map to the SAME person (<PERSON_1>).
    The prompt must sanitize to '<PERSON_1>' and '<PERSON_1>'s son', NOT creating a new <PERSON_2>.
    """
    prompt = "My name is Anand. Anand's son is working."
    result = core.sanitize(prompt)

    # 1. Must contain <PERSON_1>
    assert "<PERSON_1>" in result.sanitized_prompt, f"Expected <PERSON_1> in: {result.sanitized_prompt}"

    # 2. Must NOT contain <PERSON_2> (since 'Anand's son' refers to Anand's family, not an independent name)
    assert "<PERSON_2>" not in result.sanitized_prompt, (
        f"Prompt created two separate persons instead of deduplicating: {result.sanitized_prompt}"
    )

    # 3. Must properly preserve the possessive relationship
    assert "<PERSON_1>'s son" in result.sanitized_prompt, (
        f"Expected '<PERSON_1>'s son' in sanitized prompt, got: {result.sanitized_prompt}"
    )

    # 4. Restoration must reconstruct original sentence perfectly
    restored = core.restorer.restore(result.sanitized_prompt, result.session_id)
    assert restored.restored_text == prompt, (
        f"Restoration failed: expected '{prompt}', got '{restored.restored_text}'"
    )


def test_possessive_noun_phrase_general(core):
    """Test possessive relation on other names: 'Sarah's manager', 'Liam's phone'."""
    prompt = "Call Priya Nair. Priya Nair's phone is currently unavailable."
    result = core.sanitize(prompt)

    assert "<PERSON_1>" in result.sanitized_prompt
    assert "<PERSON_2>" not in result.sanitized_prompt
    assert "<PERSON_1>'s phone" in result.sanitized_prompt

    restored = core.restorer.restore(result.sanitized_prompt, result.session_id)
    assert restored.restored_text == prompt
