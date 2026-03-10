#!/usr/bin/env python
"""Reproduce all baseline results for Assignment 2."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.seed import set_seed


def run_command(cmd, description):
    """Run command and print output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"Error running: {cmd}")
        sys.exit(1)
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create directories
    Path("data/processed").mkdir(exist_ok=True, parents=True)
    Path("artifacts/results").mkdir(exist_ok=True, parents=True)
    
    # Step 1: Download data (if not exists)
    if not Path("data/raw/validation.json").exists():
        run_command(
            "python scripts/download_data.py --output_dir ./data/raw",
            "Downloading MTRAG dataset"
        )
    
    # Step 2: Preprocess data
    run_command(
        "python scripts/preprocess.py --config src/configs/default.yaml",
        "Preprocessing data"
    )
    
    # Step 3: Run Task A baselines
    print("\n" + "="*80)
    print("TASK A: RETRIEVAL BASELINES")
    print("="*80)
    
    # BM25
    run_command(
        "python scripts/run_baseline_task_a.py "
        "--config src/configs/retrieval/bm25.yaml "
        "--split validation "
        "--output_dir artifacts/results/task_a_bm25",
        "BM25 retrieval baseline"
    )
    
    # Dense
    run_command(
        "python scripts/run_baseline_task_a.py "
        "--config src/configs/retrieval/dense.yaml "
        "--split validation "
        "--output_dir artifacts/results/task_a_dense",
        "Dense retrieval baseline"
    )
    
    # Step 4: Evaluate Task A
    run_command(
        "python scripts/evaluate_task_a.py "
        "--predictions artifacts/results/task_a_bm25/predictions.json "
        "--ground_truth data/processed/validation.json "
        "--output_dir artifacts/results/task_a_bm25/evaluation",
        "Evaluating BM25"
    )
    
    run_command(
        "python scripts/evaluate_task_a.py "
        "--predictions artifacts/results/task_a_dense/predictions.json "
        "--ground_truth data/processed/validation.json "
        "--output_dir artifacts/results/task_a_dense/evaluation",
        "Evaluating dense retriever"
    )
    
    # Step 5: Run Task B baselines (with small test due to compute)
    print("\n" + "="*80)
    print("TASK B: GENERATION BASELINES (LIMITED EXAMPLES)")
    print("="*80)
    
    # Llama-3 (limited to 10 examples for testing)
    run_command(
        "python scripts/run_baseline_task_b.py "
        "--config src/configs/generation/llama3.yaml "
        "--split validation "
        "--max_examples 10 "
        "--output_dir artifacts/results/task_b_llama3",
        "Llama-3-8B generation (limited)"
    )
    
    print("\n" + "="*80)
    print("REPRODUCTION COMPLETE!")
    print("="*80)
    print("\nResults are in artifacts/results/")
    print("See reports/A2/ for the Assignment 2 report.")


if __name__ == "__main__":
    main()