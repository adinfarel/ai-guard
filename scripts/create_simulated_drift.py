"""
scripts/create_simulated_drift.py

Create a simulated drifted CICIDS current batch.

This is used to verify that the Evidently drift pipeline can detect distribution shift.
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path("data/processed/cicids2017/test.csv")
    output_path = Path("data/processed/cicids2017/test_drifted.csv")

    df = pd.read_csv(input_path)

    drift_multipliers = {
        "flow_duration": 10,
        "flow_packets_s": 20,
        "total_fwd_packets": 5,
        "total_backward_packets": 5,
        "packet_length_mean": 3,
    }

    for column, multiplier in drift_multipliers.items():
        if column in df.columns:
            df[column] = df[column] * multiplier
            print(f"Applied drift: {column} x {multiplier}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Simulated drift dataset saved to: {output_path}")


if __name__ == "__main__":
    main()