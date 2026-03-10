#!/usr/bin/env python
"""Download MTRAG dataset."""

import argparse
import os
import requests
import json
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/raw")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Base URL for MTRAG dataset 
    base_url = "https://github.com/IBM/mt-rag-benchmark"
    
    for split in args.splits:
        url = f"{base_url}/{split}.json"
        output_path = output_dir / f"{split}.json"
        
        print(f"Downloading {split} split...")
        try:
            download_file(url, output_path)
            print(f"Saved to {output_path}")
        except Exception as e:
            print(f"Failed to download {split}: {e}")
    
    print("Download complete!")


if __name__ == "__main__":
    main()