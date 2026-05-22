"""
tests/test_decision_engine.py

Unit tests for AI-Guard decision engine.
"""

from src.ai_guard.gateway.decision_engine import build_guard_decision

def test_guard_decision_allows_safe_prompt():
    nlp_prediction = {
        "blocked": False,
        "risk_type": "safe_prompt",
        "score": 0.01,
    }
    
    decision = build_guard_decision(nlp_prediction)
    
    assert decision["allowed"] is True
    assert decision["decision"] == "allowed"
    assert decision["blocked_by"] == []
    assert decision["reason"] == "passed_all_checks"
    assert decision["scores"]["nlp_score"] == 0.01

def test_guard_decision_blocks_unsafe_prompt():
    nlp_prediction = {
        "blocked": True,
        "risk_type": "toxic_or_jailbreak",
        "score": 0.91,
    }

    decision = build_guard_decision(nlp_prediction)

    assert decision["allowed"] is False
    assert decision["decision"] == "blocked"
    assert decision["blocked_by"] == ["nlp_firewall"]
    assert decision["reason"] == "toxic_or_jailbreak"
    assert decision["scores"]["nlp_score"] == 0.91