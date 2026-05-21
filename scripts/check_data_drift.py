"""
scripts/check_data_drift.py

Run Evidently data drift report for AI-Guard Network Firewall.

Default:
    reference = CICIDS train split
    current   = CICIDS test split

Outputs:
    reports/drift/network_drift_report.html
    reports/drift/network_drift_report.json
    reports/drift/network_drift_summary.json
"""

import argparse
import json
from pathlib import Path

from src.ai_guard.monitoring.drift import (
    build_data_drift_report,
    extract_drift_summary,
    load_csv,
    prepare_drift_frame,
    save_evidently_report,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evidently AI data drift report.")
    
    parser.add_argument(
        "--reference-path",
        type=str,
        default="data/processed/cicids2017/train.csv",
        help="Reference data path."
    )
    parser.add_argument(
        "--current-path",
        type=str,
        default="data/processed/cicids2017/test.csv",
        help="Current data path."
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default="reports/drift/network_drift_report.html",
        help="HTML report output path.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="reports/drift/network_drift_report.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="reports/drift/network_drift_summary.json",
        help="Compact summary output path.",
    )
    
    args = parser.parse_args()
    
    print(f"Loading reference data: {args.reference_path}")
    reference_data = load_csv(args.reference_path)
    
    print(f"Loading current data: {args.current_path}")
    current_data = load_csv(args.current_path)
    
    reference_feat = prepare_drift_frame(reference_data)
    current_feat = prepare_drift_frame(current_data)
    
    print("Building Evidently DataDriftPreset report...")
    report = build_data_drift_report(
        ref_data=reference_feat,
        curr_data=current_feat,
    )
    
    save_evidently_report(
        report=report,
        html_path=args.html_output,
        json_path=args.json_output,
    )
    
    summary = extract_drift_summary(
        report_json_path=args.json_output
    )
    
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved compact summary: {summary_path}")
    print("Drift summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()