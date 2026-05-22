"""
tests/test_tabular_firewall.py

Smoke tests for TabularFirewall artifact loading and inference.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.ai_guard.tabular_firewall.inference import TabularFirewall


ARTIFACT_DIR = Path("artifacts/tabular_firewall")
TEST_DATA_PATH = Path("data/processed/cicids2017/test.csv")

@pytest.mark.skipif(
    not ARTIFACT_DIR.exists() or not TEST_DATA_PATH.exists(),
    reason="Tabular artifacts or processed test data are missing."
)
def test_tabular_firewall_loads_and_predicts_sample():
    firewall = TabularFirewall.from_artifact(ARTIFACT_DIR)

    test_df = pd.read_csv(TEST_DATA_PATH)
    sample = test_df.drop(columns=["target"]).iloc[0].to_dict()

    result = firewall.predict_one(sample) #type: ignore

    assert "blocked" in result
    assert "label" in result
    assert "risk_type" in result
    assert "score" in result
    assert "threshold" in result
    assert isinstance(result["blocked"], bool)
    assert 0.0 <= result["score"] <= 1.0