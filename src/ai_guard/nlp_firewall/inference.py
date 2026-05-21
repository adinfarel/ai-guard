
"""
src/ai_guard/nlp_firewall/inference.py

Inference module for AI-Guard NLP Firewall.

This module loads a fine-tuned DistilBERT classifier and predicts whether
a user prompt is safe or should be blocked.
"""

from importlib import metadata
import json
from pathlib import Path
from typing import Any

from regex import F
import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer

class NLPFirewall:
    """
    NLP Firewall for toxic/jailbreak prompt detection.

    Responsibilities:
    - load fine-tuned tokenizer and model
    - tokenize user prompt
    - compute blocked probability
    - return allow/block decision
    """
    
    def __init__(
        self, 
        tokenizer: Any,
        model: Any,
        threshold: float = 0.5,
        max_length: int = 192,
        device: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.threshold = threshold
        self.max_length = max_length
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #type: ignore
        
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        threshold: float | None = None,
        device: str | None = None,
    ):
        """
        Load NLP firewall from artifact directory.

        Expected files:
        - tokenizer files
        - model files
        - metadata.json
        """
        if not isinstance(artifact_dir, Path):
            artifact_dir = Path(artifact_dir)
        
        if not artifact_dir.exists():
            raise FileNotFoundError(f"NLP artifact directory not found: {artifact_dir}")
        
        metadata_path = artifact_dir / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(artifact_dir))
        
        if threshold is None:
            threshold = float(metadata.get("threshold", 0.0))
        
        max_length = int(metadata.get("max_length", 192))
        
        return cls(
            tokenizer=tokenizer,
            model=model,
            threshold=threshold,
            max_length=max_length,
            device=device
        )
    
    def predict_one(self, prompt: str) -> dict[str, Any]:
        """Predict whether one user prompt should be blocked."""
        if not prompt or not prompt.strip():
            return {
                "blocked": False,
                "label": "safe",
                "risk_type": "empty_prompt",
                "score": 0.0,
                "threshold": self.threshold,
            }
        
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
        blocked_probability = float(probs[0, 1].detach().cpu().item())
        blocked = blocked_probability >= self.threshold
        
        label = "toxic_or_jailbreak" if blocked else "safe"
        risk_type = "toxic_or_jailbreak" if blocked else "safe_prompt"

        return {
            "blocked": blocked,
            "label": label,
            "risk_type": risk_type,
            "score": blocked_probability,
            "threshold": self.threshold,
        }
    
    def predict_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        """Predict multiple prompts."""
        return [self.predict_one(prompt) for prompt in prompts]