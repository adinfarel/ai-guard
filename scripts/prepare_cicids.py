'''
scripts/prepare_cicids.py

Prepare CICIDS2017 dataset for AI-Guard Network Firewall.

Input:
    data/raw/cicids2017/*.csv

Process:
    - Load all CSV files
    - Clean column names
    - Filter only BENIGN and DDoS labels
    - Convert labels:
        BENIGN -> 0
        DDoS   -> 1
    - Replace inf values with NaN
    - Drop rows with missing labels
    - Split into train/val/test

Output:
    data/processed/cicids2017/train.csv
    data/processed/cicids2017/val.csv
    data/processed/cicids2017/test.csv
    data/processed/cicids2017/metadata.json
'''

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_yaml(path: str | Path) -> dict:
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names so they are easier to use in Python.
    
    Example:
        " Destination Port" -> "destination_port"
        "Flow Bytes/s" -> "flow_bytes_s"
    """
    df = df.copy()
    
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('/', '_', regex=False)
        .str.replace('-', '_', regex=False)
    )
    
    return df

def load_all_csv(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV files from raw_dir."""
    csv_files = raw_dir.glob("*.csv")
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")
    
    frames = []
    
    for csv_file in csv_files:
        print(f"Loading: {csv_file}")
        df = pd.read_csv(csv_file)
        frames.append(df)
    
    data = pd.concat(frames, axis=0, ignore_index=True)
    
    return data

def normalize_label_value(value: object) -> str:
    """Normalize label text."""
    return str(value).strip()

def prepare_binary_cicids(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter CICIDS2017 into BENIGN vs DDoS binary classification.
    
    Output target:
        0 = BENIGN
        1 = DDoS
    """
    original_shape = df.shape
    
    label_column_raw = config['data']['label_column']
    keep_labels = config['data']['keep_labels']
    
    benign_label = config['target']['benign_label']
    attack_label = config['target']['attack_label']
    
    benign_id = config['target']['benign_id']
    attack_id = config['target']['attack_id']
    
    df = clean_column_names(df)
    
    label_column = label_column_raw.strip().lower().replace(" ", "_")
    
    if label_column not in df.columns:
        raise ValueError(
            f"Label column: '{label_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    df[label_column] = df[label_column].apply(normalize_label_value)
    
    print("\nOriginal label distribution:")
    print(df[label_column].value_counts().head(20))

    keep_labels_normalized = [normalize_label_value(label) for label in keep_labels]
    
    df = df[df[label_column].isin(keep_labels_normalized)].copy()
    
    print("\nFiltered label distribution:")
    print(df[label_column].value_counts())
    
    if df.empty:
        raise ValueError(
            "After filtering BENIGN and DDoS, dataframe is empty. "
            "Check label names in the dataset."
        )
    
    label_map = {
        benign_label: benign_id,
        attack_label: attack_id,
    }
    
    df['target'] = df[label_column].map(label_map)
    
    if df['target'].isna().any():
        bad_labels = df[df['target'].isna()][label_column].unique()
        raise ValueError(f'Some labes could not be mapped: {bad_labels}')
    
    df = df.drop(columns=[label_column])
    
    # Replace infinit values caused by rate features like Flow Bytes/s
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Keep numeric features and target
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if "target" not in numeric_cols:
        raise ValueError(f"target column is not numeric after mapping.")
    
    df = df[numeric_cols].copy()
    
    # Drop columns that are fully missing.
    df = df.dropna(axis=1, how='all')
    
    # Drop rows where target is missing
    df = df.dropna(subset=['target'])
    
    print(f"\nOriginal shape: {original_shape}")
    print(f"Prepared shape: {df.shape}")
    print(f"\nPrepared target distribution:")
    print(df['target'].value_counts())
    
    return df

def split_and_save(df: pd.DataFrame, config: dict) -> None:
    """Split dataframe into train, val, and test CSV files."""
    processed_dir = Path(config['data']['processed_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    test_size = config['split']['test_size']
    val_size = config['split']['val_size']
    random_state = config['split']['random_state']
    
    y = df['target']
    
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    # val_size is relative to total dataset.
    # Convert it into relative size of train_val
    val_relative_size = val_size / (1.0 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_df['target'],
    )
    
    train_path = processed_dir / 'train.csv'
    test_path = processed_dir / 'test.csv'
    val_path = processed_dir / 'val.csv'
    metadata_path = processed_dir / 'metadata.json'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    metadata = {
        "task": "BENIGN_vs_DDoS_binary_classification",
        "target_column": "target",
        "label_mapping": {
            "BENIGN": 0,
            "DDoS": 1,
        },
        "num_features": int(df.shape[-1] - 1),
        "num_total_rows": int(len(df)),
        "num_train_rows": int(len(train_df)),
        "num_test_rows": int(len(test_df)),
        "num_val_rows": int(len(val_df)),
        "target_distribution_total": df["target"].value_counts().to_dict(),
        "target_distribution_train": train_df["target"].value_counts().to_dict(),
        "target_distribution_val": val_df["target"].value_counts().to_dict(),
        "target_distribution_test": test_df["target"].value_counts().to_dict(),
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nSaved processed files:")
    print(f"- {train_path}")
    print(f"- {val_path}")
    print(f"- {test_path}")
    print(f"- {metadata_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CICIDS2017 BENIGN vs DDoS dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tabular_firewall.yaml",
        help="Path to tabular firewall config YAML."
    )
    
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    raw_dir = Path(config['data']['raw_dir'])
    
    raw_df = load_all_csv(raw_dir=raw_dir)
    prepared_df = prepare_binary_cicids(raw_df, config)
    split_and_save(prepared_df, config)

    print(f"CICIDS preparation complete.")

if __name__ == "__main__":
    main()