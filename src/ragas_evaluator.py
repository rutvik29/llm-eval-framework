"""RAGAS-based RAG evaluation."""
from typing import List, Dict, Any
from datasets import Dataset
import os


class RAGASEvaluator:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def evaluate(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            dataset = Dataset.from_list([{
                "question": s.get("question", ""),
                "answer": s.get("answer", ""),
                "contexts": s.get("contexts", [s.get("context","")]),
                "ground_truth": s.get("ground_truth", "")
            } for s in samples])

            result = ragas_evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
                llm=ChatOpenAI(model=self.model),
                embeddings=OpenAIEmbeddings()
            )
            return dict(result)
        except ImportError:
            return {"error": "ragas not installed", "faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}
