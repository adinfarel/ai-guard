"""
src/ai_guard/tabular_firewall/inference.py

Internal inference module for AI-Guard Network Firewall.

Important:
This module expects precomputed CICIDS-style network-flow features.
It is NOT a user-facing interface where humans manually input 78 fields.

In a real deployment, these features should come from a network telemetry
or flow feature extraction system such as CICFlowMeter.
"""

import json
from pathlib import Path
from typing import Any

from click import File
import joblib
import pandas as pd

class TabularFirewall:
    """
    Internal Network Firewall for BENIGN vs DDoS classification.

    Responsibilities:
    - load trained LightGBM pipeline
    - load expected feature columns
    - align incoming flow features
    - produce DDoS probability and block decision
    """
    
    def __init__(
        self,
        model_pipeline: Any,
        feature_columns: list[str],
        threshold: float = 0.5,
    ) -> None:
        self.model_pipeline = model_pipeline
        self.feature_columns = feature_columns
        self.threshold = threshold
    
    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        threshold: float = 0.5,
    ):
        if not isinstance(artifact_dir, Path):
            artifact_dir = Path(artifact_dir)
        
        model_path = artifact_dir / "model_pipeline.joblib"
        feature_path = artifact_dir / "feature_columns.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifacts not found: {model_path}")
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature columns not found: {feature_path}")

        model_pipeline = joblib.load(model_path)
        
        with open(feature_path, 'r', encoding='utf-8') as f:
            feature_columns = json.load(f)
        
        return cls(
            model_pipeline=model_pipeline,
            feature_columns=feature_columns,
            threshold=threshold
        )
    
    def _align_features(self, features: dict[str, Any]) -> pd.DataFrame:
        """
        Align a raw feature dict to the exact columns expected by the model.

        Missing features are filled with None.
        Extra features are ignored.

        The SimpleImputer inside the saved pipeline will handle missing values.
        """
        aligned_row = {
            column: features.get(column, None)
            for column in self.feature_columns
        }
        
        return pd.DataFrame([aligned_row], columns=self.feature_columns)

    def predict_one(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Predict whether one precomputed network flow is BENIGN or DDoS
        """
        X = self._align_features(features=features)
        
        ddos_probability = float(self.model_pipeline.predict_proba(X)[0, 1])
        blocked = ddos_probability >= self.threshold
        
        label = "DDoS" if blocked else "BENIGN"
        risk_type = "ddos" if blocked else "benign_network_flow"
        
        return {
            "blocked": blocked,
            "label": label,
            "risk_type": risk_type,
            "score": ddos_probability,
            "threshold": self.threshold,
        }
    
    def predict_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Predict multiple precomputed network flows.
        """
        return [self.predict_one(features=row) for row in rows]