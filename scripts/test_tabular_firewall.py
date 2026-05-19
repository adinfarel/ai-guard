"""
scripts/test_tabular_inference.py

Smoke test for AI-Guard TabularFirewall.

This script does not ask a human to manually input 78 CICIDS features.
Instead, it reads one sample from the processed CICIDS test split and sends it
to the internal Network Firewall inference module.
"""

import argparse

from mlflow.sagemaker import target_help
import pandas as pd

from src.ai_guard.tabular_firewall.inference import TabularFirewall

def main() -> None:
    parser = argparse.ArgumentParser(description="Test tabular firewall inference.")
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts/tabular_firewall",
        help="Path to tabular firewall artifact directory."
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/processed/cicids2017/test.csv",
        help="Path to processed CICIDS test split."
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Row index from test split to test."
    )
    
    args = parser.parse_args()
    
    firewall = TabularFirewall.from_artifact(
        artifact_dir=args.artifact_dir,
        threshold=0.5,
    )
    
    test_df = pd.read_csv(args.test_path)
    
    if args.sample_index >= len(test_df):
        raise IndexError(
            f"sample-index {args.sample_index} is out of range. "
            f"Test set has {len(test_df)} rows."
        )
    
    row = test_df.iloc[args.sample_index]
    
    true_target = row['target']
    features = row.drop(labels=["target"]).to_dict()
    
    result = firewall.predict_one(features)
    
    print("=== Tabular Firewall Smoke Test ===")
    print(f"Sample index    : {args.sample_index}")
    print(f"True target     : {true_target} ({"DDoS" if true_target == 1 else "BENIGN"})")
    print(f"Prediction      : {result}")

if __name__ == "__main__":
    main()