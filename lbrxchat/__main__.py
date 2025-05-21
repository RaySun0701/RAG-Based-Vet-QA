#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LBRXCHAT Main Entry Point
=========================

This module provides the main entry point for the LBRXCHAT application.
"""

import sys
import os
import argparse
from pathlib import Path

from lbrxchat.ui.tui import main as tui_main
from lbrxchat.core.config import (
    CORPUS_PATH, INDEX_PATH, INDEX_FILE, DEFAULT_LLM_MODEL
)


def main():
    """Main entry point for the LBRXCHAT application"""
    parser = argparse.ArgumentParser(
        description="LBRXCHAT - LIBRAXIS Advanced Chat Framework"
    )
    
    parser.add_argument(
        "--corpus", 
        type=str, 
        default=str(CORPUS_PATH),
        help="Path to the corpus file (JSONL format)"
    )
    
    parser.add_argument(
        "--index", 
        type=str, 
        default=str(INDEX_PATH),
        help="Path to the index directory"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_LLM_MODEL,
        help="Name of the LLM model to use"
    )
    
    parser.add_argument(
        "--build-index", 
        action="store_true",
        help="Build index before starting the application"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="http://localhost:1234",
        help="LM Studio host URL"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set environment variables from arguments
    if args.host:
        os.environ["LMSTUDIO_HOST"] = args.host
    
    # Override paths based on arguments
    corpus_path = Path(args.corpus)
    index_path = Path(args.index)
    index_file = index_path / "index.json"
    
    # Update environment variables
    if corpus_path != CORPUS_PATH:
        os.environ["CORPUS_PATH"] = str(corpus_path)
    
    if index_path != INDEX_PATH:
        os.environ["INDEX_PATH"] = str(index_path)
    
    # Start the TUI
    tui_main()


if __name__ == "__main__":
    main() 