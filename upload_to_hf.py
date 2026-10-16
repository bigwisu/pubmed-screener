#!/usr/bin/env python3
"""
Upload script for PubMed-Screener-420M to Hugging Face Hub.

Usage:
  huggingface-cli login
  python scripts/upload_to_hf.py --repo-id bigwisu/PubMed-Screener-420M
"""

import argparse
import os
import sys
from pathlib import Path

def upload_model(model_dir: str, repo_id: str, private: bool = False):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Error: Model directory '{model_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    
    print(f"Checking / creating repository '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"Uploading files from '{model_dir}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial release of PubMed-Screener-420M model checkpoint and model card"
    )
    print(f"Successfully uploaded PubMed-Screener-420M to https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload PubMed-Screener-420M to Hugging Face Hub")
    parser.add_argument("--model-dir", default="models/soces_pico_laya_e2e", help="Path to local model folder")
    parser.add_argument("--repo-id", default="bigwisu/PubMed-Screener-420M", help="Hugging Face repository ID")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    upload_model(model_dir=args.model_dir, repo_id=args.repo_id, private=args.private)

if __name__ == "__main__":
    main()
