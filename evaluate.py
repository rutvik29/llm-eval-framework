"""Main evaluation entry point."""
import argparse, json
from src.ragas_evaluator import RAGASEvaluator
from src.hallucination import HallucinationDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/eval_dataset.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output", default="./results/eval.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    evaluator = RAGASEvaluator(model=args.model)
    results = evaluator.evaluate(dataset)

    hallucination_detector = HallucinationDetector()
    for sample in dataset[:10]:
        h_score = hallucination_detector.score(sample.get("answer",""), sample.get("context",""))
        results["hallucination_scores"] = results.get("hallucination_scores", []) + [h_score]

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Faithfulness: {results.get('faithfulness', 0):.3f}")
    print(f"Answer Relevancy: {results.get('answer_relevancy', 0):.3f}")
    print(f"Context Recall: {results.get('context_recall', 0):.3f}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
