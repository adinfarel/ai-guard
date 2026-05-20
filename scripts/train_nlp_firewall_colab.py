"""
scripts/train_nlp_firewall_colab.py

Train AI-Guard NLP Firewall on ToxicChat using DistilBERT.

Training environment:
    Google Colab with T4 GPU recommended.

Task:
    Binary classification:
        0 = safe
        1 = toxic_or_jailbreak

Dataset:
    lmsys/toxic-chat, config toxicchat0124
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------
SEED = 42

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "lmsys/toxic-chat"
DATASET_CONFIG = "toxicchat0124"

TEXT_COLUMN = "user_input"
LABEL_COLUMNS = ["toxicity", "jailbreaking"]

MAX_LENGTH = 192
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

OUTPUT_DIR = Path("artifacts/nlp_firewall")
CHECKPOINT_DIR = Path("runs/nlp_firewall_checkpoints")

def set_seed(SEED: int = 42) -> None:
    """Set random seeds for reproducebility."""
    np.random.seed(seed=SEED)
    torch.manual_seed(seed=SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=SEED)
    
def create_blocked_label(example: dict) -> dict:
    """
    Create binary blocked label.
    
    blocked = 1 if toxicity == 1 OR jailbreaking == 1
    """
    toxicity = int(example.get('toxicity', 0))
    jailbreaking = int(example.get('jailbreaking', 0))
    
    example['labels'] = int(toxicity == 1 or jailbreaking == 1)
    
    return example

def tokenize_dataset(dataset, tokenizer: AutoTokenizer):
    """Tokenize text column."""
    def tokenize_batch(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=MAX_LENGTH,
        ) #type: ignore
    
    return dataset.map(tokenize_batch, batched=True)

def compute_metrics(eval_pred):
    """
    Compute classification metrics.
    
    Accuracy alone is not enough because unsafe prompts may be minority class.
    """
    logits, labels = eval_pred
    
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )
    
    acc = accuracy_score(labels, preds)
    
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.0
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

def main() -> None:
    set_seed(SEED)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ToxicChat Dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    
    print(dataset)
    print(f"Train columns: ", dataset['train'].column_names)
    
    train_ds = dataset['train']
    test_ds = dataset['test']
    
    print("Creating binary labels")
    train_ds = train_ds.map(create_blocked_label)
    test_ds = test_ds.map(create_blocked_label)
    
    print("Train label distribution:")
    print(train_ds.to_pandas()['labels'].value_counts()) #type: ignore
    
    print("Test label distribution:")
    print(test_ds.to_pandas()['labels'].value_counts()) #type: ignore
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenizing...")
    train_ds = tokenize_dataset(train_ds, tokenizer)
    test_ds = tokenize_dataset(test_ds, tokenizer)
    
    keep_columns = ["input_ids", "attention_mask", "labels"]
    
    train_ds = train_ds.remove_columns(
        [col for col in train_ds.column_names if col not in keep_columns]
    )
    
    test_ds = test_ds.remove_columns(
        [col for col in test_ds.column_names if col not in keep_columns]
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Training NLP Firewall...")
    trainer.train()
    
    print("Evaluating NLP Firewall...")
    metrics = trainer.evaluate()
    
    print("Final metrics:")
    print(json.dumps(metrics, indent=2))
    
    print(f"Saving NLP Firewall to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    metadata = {
        "model_type": "distilbert_sequence_classifier",
        "base_model": MODEL_NAME,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "task": "binary_prompt_firewall",
        "text_column": TEXT_COLUMN,
        "source_label_columns": LABEL_COLUMNS,
        "label_mapping": {
            "0": "safe",
            "1": "toxic_or_jailbreak",
        },
        "max_length": MAX_LENGTH,
        "threshold": 0.5,
        "metrics": metrics,
    }

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("Training complete.")

if __name__ == "__main__":
    main()