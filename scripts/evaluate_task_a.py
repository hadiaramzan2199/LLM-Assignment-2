#!/usr/bin/env python
"""Evaluate Task A retrieval predictions."""

import argparse
import json
from pathlib import Path

from src.data.loader import MTRAGDataset
from src.evaluation.metrics import evaluate_retrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    args = parser.parse_args()
    
    # Load predictions
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth
    dataset = MTRAGDataset(Path(args.ground_truth).parent, 
                          split=Path(args.ground_truth).stem)
    
    # Build ground truth mapping (only answerable questions)
    ground_truth = {}
    for ex in dataset.examples:
        # Only include answerable questions for evaluation
        if ex.answerability:  # True if answerable
            ground_truth[ex.task_id] = ex.relevant_passages
    
    print(f"Evaluating on {len(ground_truth)} answerable questions")
    
    # Filter predictions to only answerable
    filtered_preds = {tid: preds for tid, preds in predictions.items() 
                     if tid in ground_truth}
    
    # Evaluate
    metrics = evaluate_retrieval(filtered_preds, ground_truth, ks=args.ks)
    
    # Print results
    print("\n=== Retrieval Evaluation Results ===")
    for k in args.ks:
        print(f"nDCG@{k}: {metrics.get(f'ndcg@{k}', 0):.4f}")
        print(f"Precision@{k}: {metrics.get(f'precision@{k}', 0):.4f}")
        print(f"Recall@{k}: {metrics.get(f'recall@{k}', 0):.4f}")
        print(f"Hit Rate@{k}: {metrics.get(f'hit_rate@{k}', 0):.4f}")
    print(f"MRR: {metrics.get('mrr', 0):.4f}")
    
    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    detailed = {
        "num_queries": len(ground_truth),
        "num_predictions": len(filtered_preds),
        "metrics": metrics
    }
    
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump(detailed, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()