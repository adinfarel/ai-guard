"""
src/ai_guard/tabular_firewall/train.py

Train AI-Guard Network Firewall.

Task:
    BENIGN vs DDoS binary classification.

Input:
    data/processed/cicids2017/train.csv
    data/processed/cicids2017/val.csv
    data/processed/cicids2017/test.csv

Output:
    artifacts/tabular_firewall/model_pipeline.joblib
    artifacts/tabular_firewall/metrics.json
    artifacts/tabular_firewall/feature_columns.json
"""

import argparse
import json
from multiprocessing import Pipe
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn as mlsk

from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

def load_yaml(path: str | Path) -> dict:
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_split(processed_dir: str | Path, split_name: str) -> pd.DataFrame:
    """Load train/test/val split."""
    if isinstance(processed_dir, Path):
        path = processed_dir / f"{split_name}.csv"
    else:
        path = Path(processed_dir) / f"{split_name}.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {split_name}")
    
    df = pd.read_csv(path)
    print(f"Loaded {split_name}: {df.shape}")
    
    return df

def split_features_target(df: pd.DataFrame, target_col: str = 'target') -> tuple[pd.DataFrame, pd.Series]:
    """Seperate features and target."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    return X, y

def build_model_pipeline(random_state: int = 42) -> Pipeline:
    """
    Build LightGBM pipeline.

    We use SimpleImputer because CICIDS rate-based features may contain NaN
    after replacing inf values during data preparation.
    """
    # Initialized model
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model)
        ]
    )
    
    return pipeline

def evaluate_model(model: Pipeline, X, y, split_name: str, threshold: float = 0.5) -> dict:
    """Evaluate binary classifier."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        y_pred,
        average="binary",
        zero_division=0,
    )
    
    metrics = {
        "split_name": split_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    
    print(f"\n=== {split_name.upper()} METRICS ===")
    print(json.dumps(metrics, indent=2))
    
    print(f"\n=== {split_name.upper()} CLASSIFICATION REPORT ===")
    print(classification_report(y, y_pred, target_names=["BENIGN", "DDoS"]))
    
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Train AI-Guard Network Firewall.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tabular_firewall.yaml",
        help="Path to tabular firewall config YAML."
    )
    
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    processed_dir = Path(config['data']['processed_dir'])
    artifact_dir = Path(config['model']['artifact_dir'])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    random_state = config["split"]["random_state"]

    train_df = load_split(processed_dir, "train")
    val_df = load_split(processed_dir, "val")
    test_df = load_split(processed_dir, "test")
    
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)
    
    feature_columns = X_train.columns.tolist()
    
    print(f"\nNumber of features: {len(feature_columns)}")
    print("Target distribution train:")
    print(y_train.value_counts())
    
    model = build_model_pipeline(random_state=random_state)
    
    mlflow.set_experiment("ai_guard_tabular_firewall.")
    cfg_hyprprmtrs = config['hyperparameter']
    
    with mlflow.start_run(run_name="lightgbm_benign_vs_ddos"):
        mlflow.log_param("model_name", cfg_hyprprmtrs['model_name'])
        mlflow.log_param("task", cfg_hyprprmtrs['task'])
        mlflow.log_param("n_estimators", cfg_hyprprmtrs['n_estimators'])
        mlflow.log_param("learning_rate", cfg_hyprprmtrs['learning_rate'])
        mlflow.log_param("num_leaves", cfg_hyprprmtrs['num_leaves'])
        mlflow.log_param("subsample", cfg_hyprprmtrs['subsample'])
        mlflow.log_param("colsample_bytree", cfg_hyprprmtrs['colsample_bytree'])
        mlflow.log_param("class_weight", cfg_hyprprmtrs['class_weight'])
        mlflow.log_param("num_features", len(feature_columns))
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("val_rows", len(val_df))
        mlflow.log_param("test_rows", len(test_df))
        
        print(f"\nTraining LightGBM Network Firewall...")
        model.fit(X_train, y_train)
        
        val_metrics = evaluate_model(model, X_val, y_val, split_name="val")
        test_metrics = evaluate_model(model, X_test, y_test, split_name="test")
        
        for key, value in val_metrics.items():
            if key not in ['confusion_matrix', 'split']:
                mlflow.log_param(f"val_{key}", value)
                
        for key, value in test_metrics.items():
            if key not in ['confusion_matrix', 'split']:
                mlflow.log_param(f"test_{key}", value)
    
        model_path = artifact_dir / "model_pipeline.joblib"
        metrics_path = artifact_dir / "metrics.json"
        feature_path = artifact_dir / "feature_columns.json"
        
        joblib.dump(model, model_path)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "task": "BENIGN_vs_DDoS_binary_classification",
                    "model": "LightGBM",
                    "val": val_metrics,
                    "test": test_metrics
                },
                f,
                indent=2,
            )
        
        with open(feature_path, 'w', encoding='utf-8') as f:
            json.dump(feature_columns, f, indent=2)
        
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(feature_path))
        mlsk.log_model(model, artifact_path="model_pipeline")
    
    print("\nSaved artifacts")
    print(f"- {model_path}")
    print(f"- {metrics_path}")
    print(f"- {feature_path}")
    
    print(f"\nTraining complete")

if __name__ == "__main__":
    main()