"""
src/ai_guard/monitoring/drift.py

Evidently-based data drift detection for AI-Guard.

This module compares reference data and current data using Evidently's
DataDriftPreset and exports both HTML and JSON reports.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report
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
    Extract a compact drift summary from Evidently JSON output.

    Evidently JSON structure can vary slightly by version, so this function
    searches for the dataset drift metric result defensively.
    """
    report_json_path = Path(report_json_path)
    
    with open(report_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = data.get("metrics", [])
    
    drift_summary: dict[str, Any] = {
        "dataset_drift": None,
        "number_of_columns": None,
        "number_of_drifted_columns": None,
        "share_of_drifted_columns": None,
        "drifted_columns": [],
    }
    
    for metric in metrics:
        result = metric.get('result', {})
        
        if "dataset_drift" in result:
            drift_summary['dataset_drift'] = result.get('dataset_drift')
            drift_summary["number_of_columns"] = result.get("number_of_columns")
            drift_summary["number_of_drifted_columns"] = result.get("number_of_drifted_columns")
            drift_summary["share_of_drifted_columns"] = result.get("share_of_drifted_columns")

            drift_by_columns = result.get("drift_by_columns", {})
            
            drifted_columns = [
                column
                for column, column_result in drift_by_columns.items()
                if column_result.get("drift_detected") is True
            ]
            
            drift_summary['drifted_columns'] = drifted_columns
            break
    
    return drift_summary