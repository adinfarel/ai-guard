"""
src/ai_guard/gateway/decision_engine.py

Decision engine for AI-Guard Gateway.

The model returns risk scores.
The decision engine converts model outputs into gateway-level decisions.
"""

from typing import Any

def build_guard_decision(
    nlp_prediction: dict[str, Any]
) -> dict[str, Any]:
    """
    Build final gateway decision from NLP firewall prediction.

    For now, /guard only uses NLP Firewall because it is the user-facing
    prompt-level protection layer.

    Network Firewall is handled separately as an internal/background system.
    """
    blocked_by: list[str] = []
    
    if nlp_prediction.get("blocked", False):
        blocked_by.append("nlp_firewall")
    
    allowed = len(blocked_by) == 0
    
    if allowed:
        decision = "blocked"
        reason = "passed_all_checks"
    else:
        decision = "blocked"
        reason = nlp_prediction.get("risk_type", "policy_violation")
    
    return {
        "allowed": allowed,
        "decision": decision,
        "blocked_by": blocked_by,
        "reason": reason,
        "scores": {
            "nlp_score": nlp_prediction.get("score"),
        },
        "details": {
            "nlp_firewall": nlp_prediction,
        },
    }