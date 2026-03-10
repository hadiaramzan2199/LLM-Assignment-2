#!/usr/bin/env python3
"""Simple script to run BM25 baseline."""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from src.data.loader import MTRAGDataset
from src.models.retrieval.bm25 import BM25Retriever
from src.evaluation.metrics import evaluate_retrieval
from src.utils.seed import set_seed

def main():
    # Set seed
    set_seed(42)
    
    # Create output directory
    output_dir = Path("artifacts/results/task_a_bm25")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading data...")
    dataset = MTRAGDataset("data/processed", split="validation")
    print(f"Loaded {len(dataset)} examples")
    
    # Collect all passages
    all_passages = []
    for ex in dataset.examples:
        all_passages.extend(ex.corpus)
    
    # Initialize and build retriever
    print("Building BM25 index...")
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.build_index(all_passages)
    
    # Retrieve
    print("Retrieving passages...")
    predictions = retriever.retrieve_batch(
        examples=dataset.examples,
        top_k=10,
        use_history=True
    )
    
    # Save predictions
    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Build ground truth
    ground_truth = {}
    for ex in dataset.examples:
        if ex.answerability:
            ground_truth[ex.task_id] = ex.relevant_passages
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_retrieval(predictions, ground_truth, ks=[1, 3, 5, 10])
    
    # Print results
    print("\n=== RESULTS ===")
    for k in [1, 3, 5, 10]:
        print(f"nDCG@{k}: {metrics.get(f'ndcg@{k}', 0):.4f}")
        print(f"Precision@{k}: {metrics.get(f'precision@{k}', 0):.4f}")
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
