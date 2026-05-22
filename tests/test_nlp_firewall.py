"""
tests/test_nlp_firewall.py

Smoke tests for NLPFirewall artifact loading and inference.
"""

from pathlib import Path
import pytest
from src.ai_guard.nlp_firewall.inference import NLPFirewall


ARTIFACT_DIR = Path("artifacts/nlp_firewall")


@pytest.mark.skipif(
    not ARTIFACT_DIR.exists(),
    reason="NLP artifact directory is missing.",
)
def test_nlp_firewall_loads_and_predicts_prompt():
    firewall = NLPFirewall.from_artifact(
        artifact_dir=ARTIFACT_DIR,
        threshold=0.1,
        device="cpu",
    )

    result = firewall.predict_one("Tell me a short story about a cat.")

    assert "blocked" in result
    assert "label" in result
    assert "risk_type" in result
    assert "score" in result
    assert "threshold" in result
    assert isinstance(result["blocked"], bool)
    assert 0.0 <= result["score"] <= 1.0