#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LBRXCHAT Index Builder
======================

A tool for building vector indices from document corpora for LBRXCHAT.
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

from lbrxchat.core.models import MLXModelManager
from lbrxchat.core.rag import VetRAGSystem
from lbrxchat.core.config import CORPUS_PATH, INDEX_PATH


def main():
    """Build vector index from document corpus"""
    parser = argparse.ArgumentParser(
        description="LBRXCHAT Index Builder"
    )
    
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default=str(CORPUS_PATH.parent),
        help="Directory containing corpus files (JSONL format)"
    )
    
    parser.add_argument(
        "--corpus_file",
        type=str,
        default=CORPUS_PATH.name,
        help="Corpus file name within corpus_dir (JSONL format)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(INDEX_PATH),
        help="Directory to save the index"
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="nomic-embed-text-v1.5",
        help="Embedding model to use"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update paths
    corpus_path = Path(args.corpus_dir) / args.corpus_file
    output_dir = Path(args.output_dir)
    
    # Check if corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}")
        return 1
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update environment variables
    os.environ["CORPUS_PATH"] = str(corpus_path)
    os.environ["INDEX_PATH"] = str(output_dir)
    
    # Initialize model manager and RAG system
    model_manager = MLXModelManager()
    rag_system = VetRAGSystem(model_manager)
    rag_system.embedding_model = args.embedding_model
    
    # Progress callback for CLI
    progress_bar = None
    
    def update_progress(status, fraction):
        nonlocal progress_bar
        if progress_bar is None:
            progress_bar = tqdm(total=100, desc=status)
        else:
            progress_bar.desc = status
            progress_bar.n = int(fraction * 100)
            progress_bar.refresh()
    
    # Build index
    print(f"Building index from {corpus_path}...")
    success = rag_system.build_index(update_progress)
    
    if progress_bar:
        progress_bar.close()
    
    if success:
        print(f"Index built successfully and saved to {output_dir / 'index.json'}")
        return 0
    else:
        print("Error building index")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 