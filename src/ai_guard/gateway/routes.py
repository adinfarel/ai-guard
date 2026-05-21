"""
src/ai_guard/gateway/routes.py

API routes for AI-Guard Gateway.
"""

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends

from src.ai_guard.gateway.dependencies import (
    get_cicids_test_df,
    get_runtime_info,
    get_tabular_firewall,
    get_nlp_firewall,
)
from src.ai_guard.tabular_firewall.inference import TabularFirewall
from src.ai_guard.nlp_firewall.inference import NLPFirewall
from src.ai_guard.gateway.decision_engine import build_guard_decision

router = APIRouter()

@router.get("/health")
def health() -> dict[str, Any]:
    """
    Health check endpoint.
    """
    return {
        "status": "ok",
        "service": "AI-Guard Gateway",
    }

@router.get("/model-info")
def model_info(
    tabular_firewall: TabularFirewall = Depends(get_tabular_firewall),
    nlp_firewall: NLPFirewall = Depends(get_nlp_firewall),
    runtime_info: dict[str, Any] = Depends(get_runtime_info),
):
    """
    Show loaded model information.
    """
    return {
        "service": "AI-Guard Gateway",
        "runtime": runtime_info,
        "models": {
            "tabular_firewall": {
                "status": "loaded",
                "task": "BENIGN_vs_DDoS",
                "num_features": len(tabular_firewall.feature_columns), #type: ignore
                "threshold": tabular_firewall.threshold,
            },
            "nlp_firewall": {
                "status": "loaded",
                "task": "toxic_or_jailbreak_prompt_detection",
                "threshold": nlp_firewall.threshold,
                "max_length": nlp_firewall.max_length
            },
        }
    }

@router.post("/internal/check-network-sample")
def check_network_sample(
    payload: dict[str, Any],
    tabular_firewall: TabularFirewall = Depends(get_tabular_firewall),
    cicids_test_df: pd.DataFrame = Depends(get_cicids_test_df),
):
    """
    Internal testing endpoint for Network Firewall.

    Input:
        {
            "sample_index": 100
        }

    This endpoint reads one row from CICIDS test split and sends it
    to the internal tabular firewall.

    This is NOT a public endpoint for normal users.
    """
    
    sample_index = int(payload.get("sample_index", 0))
    
    if sample_index < 0 or sample_index >= len(cicids_test_df):
        raise IndexError(
            f"Sample index out of range. Valid range 0 to {len(cicids_test_df) - 1}"
        )
    
    rows = cicids_test_df.iloc[sample_index]
    
    true_target = rows['target']
    true_label = "DDoS" if true_target == 1 else "BENIGN"
    
    features = rows.drop(labels=['target']).to_dict()
    prediction = tabular_firewall.predict_one(features) #type: ignore
    
    return {
        "sample_index": sample_index,
        "true_target": true_target,
        "true_label": true_label,
        "prediction": prediction
    }

@router.post("/check-prompt")
def check_prompt(
    payload: dict[str, Any],
    nlp_firewall: NLPFirewall = Depends(get_nlp_firewall),
) -> dict[str, Any]:
    """
    User-facing endpoint for NLP Firewall.

    Input:
        {
            "prompt": "..."
        }
    """
    prompt = str(payload.get("prompt", ""))
    
    prediction = nlp_firewall.predict_one(prompt)
    
    return {
        "prompt": prompt,
        "prediction": prediction
    }

@router.post("/guard")
def guard(
    payload: dict[str, Any],
    nlp_firewall: NLPFirewall = Depends(get_nlp_firewall)
) -> dict[str, Any]:
    """
    Main user-facing AI-Guard endpoint.

    Input:
        {
            "prompt": "..."
        }

    Current behavior:
    - Runs NLP Firewall
    - Returns allow/block decision

    Future behavior:
    - If allowed, forward request to LLM backend
    - Log decision
    - Add safety metadata
    """
    prompt = str(payload.get('prompt', ''))
    
    nlp_prediction = nlp_firewall.predict_one(prompt)
    decision = build_guard_decision(nlp_prediction)
    
    return {
        "prompt": prompt,
        **decision,
    }