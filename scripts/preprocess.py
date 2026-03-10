#!/usr/bin/env python
"""Preprocess MTRAG dataset."""

import argparse
import yaml
from pathlib import Path

from src.data.loader import MTRAGDataset
from src.data.preprocessor import MTRAGPreprocessor
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/default.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config["reproducibility"]["seed"])
    
    # Initialize preprocessor
    preprocessor = MTRAGPreprocessor(
        chunk_size=config["data"]["chunk_size"],
        chunk_overlap=config["data"]["chunk_overlap"],
        min_chunk_length=config["data"]["min_chunk_length"]
    )
    
    # Process each split
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        
        # Load raw data
        raw_dir = Path(config["data"]["raw_dir"])
        dataset = MTRAGDataset(raw_dir, split=split)
        print(f"Loaded {len(dataset)} examples")
        
        # Process
        processed = preprocessor.process_dataset(dataset)
        
        # Save
        processed_dir = Path(config["data"]["processed_dir"])
        processed_dir.mkdir(exist_ok=True, parents=True)
        output_path = processed_dir / f"{split}.json"
        
        preprocessor.save_processed(processed, output_path)
        print(f"Saved processed data to {output_path}")
        
        # Build passage lookup
        lookup = preprocessor.build_passage_lookup(processed)
        lookup_path = processed_dir / f"{split}_passage_lookup.json"
        import json
        with open(lookup_path, 'w') as f:
            json.dump(lookup, f)
        print(f"Saved passage lookup to {lookup_path}")


if __name__ == "__main__":
    main()