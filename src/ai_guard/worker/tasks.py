"""
src/ai_guard/worker/tasks.py

Celery background tasks for AI-Guard.
"""
from pathlib import Path

from src.ai_guard.monitoring.drift import (
    build_data_drift_report,
    extract_drift_summary,
    load_csv,
    prepare_drift_frame,
    save_evidently_report,
)
from src.ai_guard.worker.celery_app import celery_app

@celery_app.task(name="ai_guard.run_network_drift_check")
def run_network_drift_check(
    reference_path: str = "data/processed/cicids2017/train.csv",
    current_path: str = "data/processed/cicids2017/test.csv",
    html_output: str = "reports/drift/network_drift_report_async.html",
    json_output: str = "reports/drift/network_drift_report_async.json",
    summary_output: str = "reports/drift/network_drift_summary_async.json",
) -> dict:
    """
    Run Evidently network data drift report in the background.

    Returns compact drift summary.
    """
    reference_df = load_csv(reference_path)
    current_df = load_csv(current_path)
    
    reference_feat = prepare_drift_frame(reference_df)
    current_feat = prepare_drift_frame(current_df)
    
    report = build_data_drift_report(
        ref_data=reference_feat,
        curr_data=current_feat,
    )
    
    save_evidently_report(
        report=report,
        html_path=html_output,
        json_path=json_output,
    )
    
    summary = extract_drift_summary(json_output)
    
    summary_path = Path(summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    
    return {
        "status": "completed",
        "reference_path": reference_path,
        "current_path": current_path,
        "html_output": html_output,
        "json_output": json_output,
        "summary_output": summary_output,
        "summary": summary,
    }