"""
src/ai_guard/gateway/app.py

FastAPI application entrypoint for AI-Guard Gateway.

Modern architecture:
- lifespan for startup/shutdown
- app.state for runtime resources
- APIRouter for endpoints
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI

from src.ai_guard.nlp_firewall.inference import NLPFirewall
from src.ai_guard.gateway.routes import router
from src.ai_guard.tabular_firewall.inference import TabularFirewall

APP_NAME = "AI-Guard Gateway"
APP_VERSION = "0.1.0"

TABULAR_ARTIFACT_DIR = Path("artifacts/tabular_firewall")
CICIDS_TEST_PATH = Path("data/processed/cicids2017/test.csv")
NLP_ARTIFACT_DIR = Path("artifacts/nlp_firewall")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Load runtime resources once when the application starts.

    Resources loaded here:
    - trained TabularFirewall artifact
    - CICIDS test split for internal demo endpoint
    """
    print("[startup] Loading AI-Guard runtime artifacts...")
    
    tabular_firewall = TabularFirewall.from_artifact(
        artifact_dir=TABULAR_ARTIFACT_DIR,
        threshold=0.5,
    )
    
    nlp_firewall = NLPFirewall.from_artifact(
        artifact_dir=NLP_ARTIFACT_DIR,
        threshold=0.1,
        device="cpu"
    )
    
    if not CICIDS_TEST_PATH.exists():
        raise FileNotFoundError(f"CICIDS test split not found: {CICIDS_TEST_PATH}")
    
    cicids_test_df = pd.read_csv(CICIDS_TEST_PATH)
    
    app.state.tabular_firewall = tabular_firewall
    app.state.nlp_firewall = nlp_firewall
    app.state.cicids_test_df = cicids_test_df
    app.state.runtime_info = {
        "tabular_artifact_dir": str(TABULAR_ARTIFACT_DIR),
        "nlp_artifact_dir": str(NLP_ARTIFACT_DIR),
        "cicids_test_path": str(CICIDS_TEST_PATH),
        "num_cicids_test_rows": len(cicids_test_df),
    }
    
    print("[startup] Runtime artifacts loaded.")
    
    yield
    
    print("[shutdown] Cleaning AI-Guard runtime artifacts...")
    
    app.state.tabular_firewall = None
    app.state.nlp_firewall = None
    app.state.cicids_test_df = None
    app.state.runtime_info = None
    
    print("[shutdown] Cleanup complete.")
    
def create_app() -> FastAPI:
    """
    Application factory
    
    This makes the app easier to test and extend later.
    """
    app = FastAPI(
        title=APP_NAME,
        description="Production-style AI security gateway for LLM API protection.",
        version=APP_VERSION,
        lifespan=lifespan
    )
    
    app.include_router(
        router
    )
    
    return app

app = create_app()