#!/usr/bin/env python
"""Run Task A retrieval baseline."""

import argparse
import json
import yaml
from pathlib import Path

from src.data.loader import MTRAGDataset
from src.models.retrieval.bm25 import BM25Retriever
from src.models.retrieval.dense_retriever import DenseRetriever
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get("reproducibility", {}).get("seed", 42))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load processed data
    processed_dir = Path("data/processed")
    dataset = MTRAGDataset(processed_dir, split=args.split)
    print(f"Loaded {len(dataset)} examples for {args.split} split")
    
    # Collect all corpus passages
    all_passages = []
    for ex in dataset.examples:
        all_passages.extend(ex.corpus)
    
    # Initialize retriever
    method = config["retrieval"]["method"]
    print(f"Using {method} retrieval")
    
    if method == "bm25":
        retriever = BM25Retriever(
            k1=config["retrieval"]["bm25"]["k1"],
            b=config["retrieval"]["bm25"]["b"]
        )
    elif method == "dense":
        retriever = DenseRetriever(
            model_name=config["retrieval"]["dense"]["model_name"],
            batch_size=config["retrieval"]["dense"]["batch_size"]
        )
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    
    # Build index
    retriever.build_index(all_passages)
    
    # Save index if configured
    if config.get("output", {}).get("save_index", False):
        index_dir = Path(config["output"]["index_dir"])
        retriever.save(index_dir)
        print(f"Saved index to {index_dir}")
    
    # Retrieve for all examples
    predictions = retriever.retrieve_batch(
        examples=dataset.examples,
        top_k=config["retrieval"]["top_k"],
        use_history=config["retrieval"]["use_history"]
    )
    
    # Save predictions
    pred_path = output_dir / "predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {pred_path}")
    
    # Also save in format expected by evaluator
    formatted_preds = []
    for task_id, passages in predictions.items():
        formatted_preds.append({
            "task_id": task_id,
            "retrieved_passages": passages
        })
    
    formatted_path = output_dir / "task_a_predictions.json"
    with open(formatted_path, 'w') as f:
        json.dump(formatted_preds, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()