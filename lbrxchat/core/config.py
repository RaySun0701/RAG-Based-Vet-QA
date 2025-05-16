#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LBRXCHAT Configuration
======================

System configuration and constants for LBRXCHAT.
"""

import os
from pathlib import Path

# === Paths and Directories ===
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
CORPUS_PATH = PROJECT_ROOT / "corpus" / "data.jsonl"
INDEX_PATH = PROJECT_ROOT / "indexes"
INDEX_FILE = INDEX_PATH / "index.json"
MLX_MODELS_PATH = Path.home() / ".cache" / "mlx-lm"

# Ensure directories exist
INDEX_PATH.mkdir(parents=True, exist_ok=True)

# === Model Settings ===
MODEL_TTL = 30  # Minutes before unloading model

# Default LLM model
DEFAULT_LLM_MODEL = "qwen3-8b-mlx"  # Can be changed by user
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v1.5"  # Embedding model

# System prompts for different models
SYSTEM_PROMPTS = {
    "qwen3-8b-mlx": "You are a helpful assistant. Based only on the given context, answer the user's question. If you don't know the answer from the context, admit it.",
    "qwen3-14b-mlx": "You are a helpful assistant. Based only on the given context, answer the user's question. If you don't know the answer from the context, admit it.",
    "deepcogito-cogito-v1-preview-qwen-32b": "You are a helpful assistant. Based only on the given context, answer the user's question. If you don't know the answer from the context, admit it.",
    "llama3-8b-mlx": "You are a helpful assistant. Based only on the given context, answer the user's question. If you don't know the answer from the context, admit it."
}

# === LM Studio Settings ===
LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://localhost:1234")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "not-needed")  # Usually not needed

# === RAG Settings ===
DEFAULT_TOP_K = 5  # Number of context documents to retrieve
EMBEDDING_DIMENSION = 768  # Dimension of embedding vectors
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for retrieval

# === UI Settings ===
UI_THEME = "dark"  # dark or light
MAX_CHAT_HISTORY = 10  # Maximum number of messages to retain
CHAT_TEMPERATURE = 0.2  # Temperature for LLM generation

# === Debug Settings ===
DEBUG = os.environ.get("LBRXCHAT_DEBUG", "0") == "1"
LOG_LEVEL = os.environ.get("LBRXCHAT_LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "lbrxchat.log" 