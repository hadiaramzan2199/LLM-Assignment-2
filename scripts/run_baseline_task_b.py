#!/usr/bin/env python
"""Run Task B generation baseline."""

import argparse
import json
import yaml
from pathlib import Path

import torch

from src.data.loader import MTRAGDataset
from src.models.generation.llm_generator import LLMGenerator, GenerationConfig
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Limit examples for testing")
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
    
    # Load passage lookup
    lookup_path = processed_dir / f"{args.split}_passage_lookup.json"
    with open(lookup_path, 'r') as f:
        passage_lookup = json.load(f)
    
    # Limit examples if specified
    examples = dataset.examples
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Limited to {len(examples)} examples")
    
    # Initialize generator
    gen_config = GenerationConfig(
        model_name=config["generation"]["model_name"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        top_k=config["generation"]["top_k"],
        do_sample=config["generation"]["do_sample"],
        num_beams=config["generation"]["num_beams"],
        repetition_penalty=config["generation"]["repetition_penalty"],
        use_history=config["generation"]["use_history"]
    )
    
    generator = LLMGenerator(
        config=gen_config,
        quantization=config["generation"].get("quantization", "4bit"),
        device_map=config["generation"].get("device_map", "auto")
    )
    
    # Generate responses
    print("Generating responses...")
    predictions = generator.generate_batch(
        examples=examples,
        passage_lookup=passage_lookup,
        use_history=config["generation"]["use_history"]
    )
    
    # Save predictions
    pred_path = output_dir / "predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved predictions to {pred_path}")
    
    # Format for evaluation
    formatted_preds = []
    for task_id, response in predictions.items():
        formatted_preds.append({
            "task_id": task_id,
            "response": response
        })
    
    formatted_path = output_dir / "task_b_predictions.json"
    with open(formatted_path, 'w') as f:
        json.dump(formatted_preds, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()