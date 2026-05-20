"""
src/ai_guard/gateway/routes.py

API routes for AI-Guard Gateway.
"""

from curses.ascii import HT
from random import sample
from typing import Any

from fastapi.exceptions import DependencyScopeError
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.ai_guard.gateway.dependencies import (
    get_cicids_test_df,
    get_runtime_info,
    get_tabular_firewall,
)
from src.ai_guard.tabular_firewall.inference import TabularFirewall

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