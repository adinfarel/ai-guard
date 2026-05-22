"""
src/ai_guard/monitoring/drift.py

Evidently-based data drift detection for AI-Guard.

This module compares reference data and current data using Evidently's
DataDriftPreset and exports both HTML and JSON reports.
"""

import json
from pathlib import Path
from re import I
from typing import Any

from numpy import isin
import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report
from torch import threshold
# from evidently.render.html import HTMLRender #type: ignore

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load csv file into DataFrame."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    return pd.read_csv(path)

def prepare_drift_frame(
    df: pd.DataFrame,
    target_column: str = "target"
) -> pd.DataFrame:
    """
    Prepare dataframe for drift detection.

    We remove target column because feature drift should compare input features,
    not labels.
    """
    df = df.copy()
    
    if target_column in df.columns:
        df = df.drop(columns=[target_column])
        
    return df

def build_data_drift_report(
    ref_data: pd.DataFrame,
    curr_data: pd.DataFrame,
) -> Report:
    """Build Evidently Data Drift report."""
    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )
    
    my_report = report.run(
        reference_data=ref_data,
        current_data=curr_data,
    )
    
    return my_report #type: ignore

def save_evidently_report(
    report: Report,
    html_path: str | Path,
    json_path: str | Path,
) -> None:
    """
    Save Evidently report as HTML and JSON.

    HTML is useful for visual inspection.
    JSON is useful for automation and CI/reporting.
    """
    html_path = Path(html_path)
    json_path = Path(json_path)
    
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    report.save_html(str(html_path)) #type: ignore
    
    report_json = report.json() #type: ignore
    
    if isinstance(report_json, str):
        parsed_json = json.loads(report_json)
    else:
        parsed_json = report_json
        
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=2)
    
    print(f"Saved HTML report: {html_path}")
    print(f"Saved JSON report: {json_path}")

def extract_drift_summary(report_json_path: str | Path) -> dict[str, Any]:
    """
    Extract compact drift summary from Evidently metric v2 JSON output.

    Supported Evidently structure example:
    {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "value": {"count": 5.0, "share": 0.064}
            },
            {
                "metric_name": "ValueDrift(column=flow_duration,...threshold=0.1)",
                "config": {
                    "column": "flow_duration",
                    "threshold": 0.1
                },
                "value": 4.63
            }
        ]
    }
    """
    report_json_path = Path(report_json_path)
    
    with open(report_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = data.get("metrics", [])
    
    drifted_columns: list[str] = []
    value_drifted_metrics: dict[str, dict[str, Any]] = {}
    
    number_of_drifted_columns: int | None = None
    share_of_drifted_columns: float | None = None
    
    for metric in metrics:
        metric_name = str(metric.get("metric_name"))
        config = metric.get("config", {})
        value = metric.get("value", {})
        
        if metric_name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                count = value.get("count")
                share = value.get("share")
                
                if count is not None:
                    number_of_drifted_columns = int(count)
                
                if share is not None:
                    share_of_drifted_columns = float(share)
        
        elif metric_name.startswith("ValueDrift"):
            column = config.get("column")
            threshold = config.get("threshold")
            
            if column is None:
                continue
            
            if threshold is None:
                threshold = 0.1
            
            try:
                drift_value = float(value)
                drift_threshold = float(threshold)
            except (TypeError, ValueError):
                continue
            
            is_drifted = drift_value >= drift_threshold
            
            value_drifted_metrics[column] = {
                "drift_score": drift_value,
                "threshold": threshold,
                "drifted": is_drifted,
            }
            
            if is_drifted:
                drifted_columns.append(column)
        
    number_of_columns = len(value_drifted_metrics)
    
    if number_of_drifted_columns is None:
        number_of_drifted_columns = len(drifted_columns)
    
    if share_of_drifted_columns is None:
        if number_of_columns > 0:
            share_of_drifted_columns = number_of_drifted_columns / number_of_columns
        else:
            share_of_drifted_columns = 0.0
            
    dataset_drift = number_of_drifted_columns > 0
        
    return {
        "parser_status": "found_metric_drift",
        "dataset_drift": dataset_drift,
        "number_of_columns": number_of_columns,
        "number_of_drifted_columns": number_of_drifted_columns,
        "share_of_drifted_columns": share_of_drifted_columns,
        "drifted_columns": sorted(set(drifted_columns)),
        "value_drift_metrics": value_drifted_metrics,
    }