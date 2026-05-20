"""
src/ai_guard/gateway/dependencies.py

FastAPI dependencies for accessing runtime resources.

Runtime resources are stored in app.state during lifespan startup.
"""

from typing import Any

import pandas as pd
from fastapi import Request, HTTPException, status

from src.ai_guard.tabular_firewall.inference import TabularFirewall

def get_tabular_firewall(request: Request) -> TabularFirewall:
    """
    Get loaded TabularFirewall from app.state
    """
    
    firewall = getattr(request.app.state, "tabular_firewall", None)
    
    if firewall is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tabular firewall is not loaded."
        )
    
    return firewall

def get_cicids_test_df(request: Request) -> pd.DataFrame:
    """
    Get loaded CICIDS test dataframe from app.state
    """
    
    test_df = getattr(request.app.state, "cicids_test_df", None)
    
    if test_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CICIDS test data is not loaded."
        )
    
    return test_df

def get_runtime_info(request: Request) -> dict[str, Any]:
    """
    Get runtime metadata from app.state
    """
    runtime_info = getattr(request.app.state, "runtime_info", None)
    
    if runtime_info is None:
        return {}

    return runtime_info