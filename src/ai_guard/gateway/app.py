"""
src/ai_guard/gateway/app.py

FastAPI gateway for AI-Guard.

Current endpoints:
- GET  /health
- GET  /model-info
- POST /internal/check-network-sample

Important:
The network firewall uses precomputed CICIDS-style network-flow features.
It is exposed here only as an internal/testing endpoint.
"""

from curses.ascii import HT
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.ai_guard.tabular_firewall.inference import TabularFirewall

# -----------------------
APP_NAME = "AI-Guard Gateway"
TABULAR_ARTIFACT_DIR = Path("artifacts/tabular_firewall")
CICIDS_TEST_PATH = Path("data/processed/cicids2017/test.csv")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    global tabular_firewall
    global cicids_test_df
    
    tabular_firewall = TabularFirewall.from_artifact(
        artifact_dir=TABULAR_ARTIFACT_DIR,
        threshold=0.5,
    )
    
    if not CICIDS_TEST_PATH.exists():
        raise FileNotFoundError(f"CICIDS test split not found: {CICIDS_TEST_PATH}")
    
    cicids_test_df = pd.read_csv(CICIDS_TEST_PATH)
    
    yield
    
app = FastAPI(
    title=APP_NAME,
    description="Production-Style AI Security Gateway for LLM API Protection.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

tabular_firewall: TabularFirewall | None = None
cicids_test_df: pd.DataFrame | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": APP_NAME,
    }

@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Show loaded model information."""
    
    if tabular_firewall is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tabular firewall is not loaded."
        )
    
    return {
        "service": APP_NAME,
        "models": {
            "tabular_firewall": {
                "status": "loaded",
                "task": "BENIGN_vs_DDoS",
                "num_features": len(tabular_firewall.feature_columns),
                "threshold": tabular_firewall.threshold,
            },
            "nlp_firewall": {
                "status": "not_loaded_yet",
                "task": "toxic_or_jailbreak_prompt_detection",
            },
        },
    }

@app.post("/internal/check-network-sample")
def check_network_sample(payload: dict[str, Any]) -> dict[str, Any]:
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
    if tabular_firewall is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tabular firewall is not loaded."
        )
    
    if cicids_test_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CICIDS test data is not laoded."
        )
    
    sample_index = int(payload.get("sample_index", 0))
    
    if sample_index < 0 or sample_index >= len(cicids_test_df):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sample index out of range. Valid range 0 to {len(cicids_test_df) - 1}"
        )
    
    row = cicids_test_df.iloc[sample_index]
    
    true_target = int(row["target"])
    true_label = "DDoS" if true_target == 1 else "BENIGN"

    features = row.drop(labels=["target"]).to_dict()
    prediction = tabular_firewall.predict_one(features) #type: ignore
    
    return {
        "sample_index": sample_index,
        "true_target": true_target,
        "true_label": true_label,
        "prediction": prediction,
    }
