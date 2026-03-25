"""NLI-based hallucination detection."""
from typing import Optional


class HallucinationDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def score(self, answer: str, context: str) -> float:
        if not self.model or not answer or not context:
            return 0.5
        try:
            sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
            if not sentences:
                return 1.0
            scores = self.model.predict([(context, s) for s in sentences])
            entailment_scores = [max(0, s[1]) for s in scores] if hasattr(scores[0], '__iter__') else [max(0, s) for s in scores]
            return sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.5
        except Exception:
            return 0.5
