#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLX Veterinary RAG TUI System
=============================

A sophisticated TUI-based RAG system for veterinary Q&A using MLX models natively.
Models are loaded JIT (Just In Time) with a 30-minute TTL.

Usage:
    python mlx_vet_rag_tui.py

Requirements:
    - mlx-lm
    - textual
    - numpy
    - scikit-learn
    - rich
    - lmstudio
"""

import os
import sys
import json
import time
import asyncio
import threading
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import subprocess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Header, Footer, Button, Static, Input, Label,
    Select, OptionList, Log, RichLog,
)
from textual.reactive import reactive
from textual.binding import Binding
from textual import events
from textual.widgets.option_list import Option

# LM Studio and Requests for API access
import requests
try:
    from lmstudio import Client as LMStudioClient
    LMSTUDIO_NATIVE_API = True
except ImportError:
    LMSTUDIO_NATIVE_API = False

# === Constants ===
PROJECT_ROOT = Path(__file__).parent.absolute()
CORPUS_PATH = PROJECT_ROOT / "vet_corpus" / "chunked_merck.jsonl"
INDEX_PATH = PROJECT_ROOT / "vet_index_mlx_native"
INDEX_FILE = INDEX_PATH / "index.json"
MLX_MODELS_PATH = Path.home() / ".cache" / "mlx-lm"
MODEL_TTL = 30  # Minutes before unloading model

# Default MLX model to use
DEFAULT_LLM_MODEL = "qwen3-8b-mlx"  # Can be changed by user
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v1.5"  # Embedding model

# System prompts for different models
SYSTEM_PROMPTS = {
    "qwen3-8b-mlx": "You are a helpful veterinary assistant. Based only on the given context, answer the user's question about animal health and care. If you don't know the answer from the context, admit it.",
    "qwen3-14b-mlx": "You are a helpful veterinary assistant. Based only on the given context, answer the user's question about animal health and care. If you don't know the answer from the context, admit it.",
    "deepcogito-cogito-v1-preview-qwen-32b": "You are a helpful veterinary assistant. Based only on the given context, answer the user's question about animal health and care. If you don't know the answer from the context, admit it."
}

# === Model Management ===
class MLXModelManager:
    """Manages MLX model loading, serving, and TTL lifecycle"""
    
    def __init__(self, models_path: Path = MLX_MODELS_PATH):
        self.models_path = models_path
        self.active_servers = {}  # type: Dict[str, Dict[str, Any]]
        self.models_info = {}  # type: Dict[str, Dict[str, Any]]
        self.server_ports = {}  # type: Dict[str, int]
        self.next_port = 8000
        self.lock = threading.Lock()
        
        # Initialize LM Studio native client if available
        self.lmstudio_client = None
        if LMSTUDIO_NATIVE_API:
            try:
                self.lmstudio_client = LMStudioClient()
                print("LM Studio native API initialized")
            except Exception as e:
                print(f"Error initializing LM Studio native API: {e}")
                self.lmstudio_client = None
        
        # Ensure the models directory exists
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _get_next_port(self) -> int:
        """Get the next available port"""
        with self.lock:
            port = self.next_port
            self.next_port += 1
            return port
    
    def _cleanup_loop(self):
        """Periodically check and clean up expired models"""
        while True:
            time.sleep(60)  # Check every minute
            self._cleanup_expired_models()
    
    def _cleanup_expired_models(self):
        """Unload models that have exceeded their TTL"""
        now = datetime.now()
        with self.lock:
            expired_models = []
            for model_name, info in self.active_servers.items():
                last_used = info.get("last_used", now)
                ttl_minutes = info.get("ttl_minutes", MODEL_TTL)
                
                if now - last_used > timedelta(minutes=ttl_minutes):
                    expired_models.append(model_name)
            
            for model_name in expired_models:
                self.unload_model(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of models available in LM Studio"""
        models = []
        
        # Try native API first
        if self.lmstudio_client:
            try:
                # Native API doesn't have a direct method to list all models
                # Use REST API as fallback for now
                response = requests.get("http://localhost:1234/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["id"] for model in data.get("data", [])]
            except Exception as e:
                print(f"Error fetching models via native API: {e}")
        
        # Fallback to REST API
        if not models:
            try:
                response = requests.get("http://localhost:1234/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["id"] for model in data.get("data", [])]
            except Exception as e:
                print(f"Error fetching models via REST API: {e}")
                models = [DEFAULT_LLM_MODEL]  # Fallback to default
        
        return models
    
    def load_model(self, model_name: str, ttl_minutes: int = MODEL_TTL) -> int:
        """
        Load model and return server port
        """
        with self.lock:
            # If model is already loaded, update last_used and return port
            if model_name in self.active_servers:
                server_info = self.active_servers[model_name]
                server_info["last_used"] = datetime.now()
                return server_info["port"]
            
            # For this simulation, we'll use the existing LM Studio server on port 1234
            # In a real implementation, we would start mlx-serve here
            self.active_servers[model_name] = {
                "port": 1234,
                "last_used": datetime.now(),
                "ttl_minutes": ttl_minutes,
                "process": None
            }
            
            return 1234
    
    def unload_model(self, model_name: str) -> bool:
        """Unload model by stopping its server"""
        with self.lock:
            if model_name not in self.active_servers:
                return False
            
            server_info = self.active_servers[model_name]
            # We don't actually need to stop anything since we're using LM Studio
            
            del self.active_servers[model_name]
            return True
    
    def update_last_used(self, model_name: str):
        """Update the last used timestamp for a model"""
        with self.lock:
            if model_name in self.active_servers:
                self.active_servers[model_name]["last_used"] = datetime.now()


# === RAG System ===
class VetRAGSystem:
    """Core RAG system for veterinary question answering"""
    
    def __init__(self, model_manager: MLXModelManager):
        self.model_manager = model_manager
        self.index_data = None
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.llm_model = DEFAULT_LLM_MODEL
        self.chat_history = []
        
        # Ensure index directory exists
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
    
    def load_corpus(self, callback: Optional[Callable[[str, float], None]] = None) -> List[Dict[str, Any]]:
        """Load corpus of veterinary documents"""
        documents = []
        
        if not CORPUS_PATH.exists():
            if callback:
                callback("Corpus file not found", 1.0)
            return []
        
        total_lines = sum(1 for _ in open(CORPUS_PATH, 'r', encoding='utf-8'))
        
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                documents.append(doc)
                if callback:
                    callback(f"Loading documents: {i+1}/{total_lines}", (i+1)/total_lines)
        
        return documents
    
    def build_index(self, callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Build vector index from corpus"""
        if callback:
            callback("Loading documents...", 0.0)
        
        documents = self.load_corpus(callback)
        if not documents:
            return False
        
        if callback:
            callback("Extracting document content...", 0.1)
        
        texts = [doc["content"] for doc in documents]
        
        # In a real implementation, we would generate embeddings here
        # For now, we'll simulate it with random embeddings
        if callback:
            callback("Generating embeddings (simulated)...", 0.2)
        
        # Generate 768-dimensional embeddings
        embeddings = [np.random.rand(768).tolist() for _ in range(len(texts))]
        
        # Save embeddings to index
        if callback:
            callback("Saving index...", 0.8)
        
        index_data = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            index_data.append({
                "id": i,
                "content": doc["content"],
                "metadata": {
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "section": doc.get("section", ""),
                    "article_id": doc.get("article_id", ""),
                    "chunk_id": doc.get("chunk_id", 0)
                },
                "embedding": embedding
            })
        
        with open(INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(index_data, f)
        
        self.index_data = index_data
        
        if callback:
            callback("Index built successfully", 1.0)
        
        return True
    
    def load_index(self, callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Load existing vector index"""
        if not INDEX_FILE.exists():
            if callback:
                callback("Index not found", 1.0)
            return False
        
        if callback:
            callback("Loading index...", 0.2)
        
        try:
            with open(INDEX_FILE, 'r', encoding='utf-8') as f:
                self.index_data = json.load(f)
            
            if callback:
                callback(f"Index loaded with {len(self.index_data)} documents", 1.0)
            
            return True
        except Exception as e:
            if callback:
                callback(f"Error loading index: {e}", 1.0)
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity"""
        if not self.index_data:
            if not self.load_index():
                return []
        
        # In a real implementation, we would generate a query embedding here
        # For demonstration, we'll simulate similarity with random values
        similarities = np.random.rand(len(self.index_data))
        
        # Get top k documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                "content": self.index_data[idx]["content"],
                "metadata": self.index_data[idx]["metadata"],
                "score": float(similarities[idx])
            })
        
        return results
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using MLX model"""
        # Get system prompt for model
        system_prompt = SYSTEM_PROMPTS.get(self.llm_model, SYSTEM_PROMPTS[DEFAULT_LLM_MODEL])
        
        # Prepare context
        context_text = "\n\n".join([doc["content"] for doc in contexts])
        
        # Construct messages
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context_text}"},
            *self.chat_history[-6:],  # Last 3 exchanges (6 messages)
            {"role": "user", "content": query}
        ]
        
        # Update model last used time
        self.model_manager.update_last_used(self.llm_model)
        
        # Try to use native LM Studio API if available
        if self.model_manager.lmstudio_client:
            try:
                # Get model handle
                model = self.model_manager.lmstudio_client.llm.model(self.llm_model)
                
                # Generate response
                response = model.chat.completions.create(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024
                )
                
                answer = response.choices[0].message.content
                
                # Update chat history
                self.chat_history.append({"role": "user", "content": query})
                self.chat_history.append({"role": "assistant", "content": answer})
                
                # Keep chat history to last 5 exchanges (10 messages)
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]
                
                return {
                    "answer": answer,
                    "sources": contexts,
                    "success": True
                }
            except Exception as e:
                print(f"Error using native API: {e}")
                # Fall back to REST API
        
        # Fallback to REST API
        try:
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1024
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                
                # Update chat history
                self.chat_history.append({"role": "user", "content": query})
                self.chat_history.append({"role": "assistant", "content": answer})
                
                # Keep chat history to last 5 exchanges (10 messages)
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]
                
                return {
                    "answer": answer,
                    "sources": contexts,
                    "success": True
                }
            else:
                return {
                    "answer": f"Error: {response.status_code} - {response.text}",
                    "sources": [],
                    "success": False
                }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def ask(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Ask a question and get answer with sources"""
        # Search for relevant contexts
        contexts = self.search(query, top_k=top_k)
        
        # Generate answer
        result = self.generate_answer(query, contexts)
        
        return result


# === TUI Application ===
class VetRAGApp(App):
    """TUI application for veterinary RAG QA system"""
    
    TITLE = "Veterinary RAG System with MLX"
    SUB_TITLE = "Powered by MLX and Textual"
    
    CSS_PATH = "mlx_vet_rag_tui.css"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload_index", "Reload Index"),
        Binding("c", "clear_chat", "Clear Chat"),
        Binding("m", "change_model", "Change Model"),
    ]
    
    def __init__(self):
        super().__init__()
        self.model_manager = MLXModelManager()
        self.rag_system = VetRAGSystem(self.model_manager)
        self.available_models = self.model_manager.get_available_models()
    
    def compose(self) -> ComposeResult:
        """Create child widgets"""
        # Create header
        yield Header()
        
        # Main layout
        with Container(id="app-grid"):
            # Left panel - Settings and Info
            with Container(id="left-panel"):
                yield Label("Models", id="models-header")
                yield Select(
                    [(model, model) for model in self.available_models],
                    id="model-selector",
                    value=DEFAULT_LLM_MODEL,
                    prompt="Select LLM Model"
                )
                
                yield Button("Build/Reload Index", id="build-index-btn")
                yield Button("Clear Chat", id="clear-chat-btn")
                
                yield Label("System Information", id="system-info-header")
                yield Static(id="system-info")
                
                # Progress bar for loading
                yield Static("", id="progress-label")
                yield Static("", id="progress-bar")
            
            # Right panel - Chat and Results
            with Container(id="right-panel"):
                # Chat history
                yield RichLog(id="chat-log", markup=True, highlight=True)
                
                # Input area
                with Container(id="input-container"):
                    yield Input(placeholder="Ask a veterinary question...", id="query-input")
                    yield Button("Send", id="send-btn")
        
        # Footer
        yield Footer()
    
    def on_mount(self) -> None:
        """When app is mounted (started)"""
        # Update system info
        self.update_system_info()
        
        # Load index if exists
        self.load_index_async()
    
    def update_system_info(self) -> None:
        """Update system information display"""
        info_widget = self.query_one("#system-info", Static)
        
        # Check if index exists
        index_status = "Loaded" if self.rag_system.index_data else "Not loaded"
        if INDEX_FILE.exists():
            index_size = INDEX_FILE.stat().st_size // 1024  # KB
            index_status = f"Available ({index_size} KB)"
        
        # Active models
        active_models = list(self.model_manager.active_servers.keys())
        active_models_str = ", ".join(active_models) if active_models else "None"
        
        # LM Studio API status
        api_status = "Available" if self.model_manager.lmstudio_client else "Not available"
        
        # Build info table
        table = Table(box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        
        table.add_row("Current LLM", self.rag_system.llm_model)
        table.add_row("Index Status", index_status)
        table.add_row("Active Models", active_models_str)
        table.add_row("Model TTL", f"{MODEL_TTL} minutes")
        table.add_row("LM Studio API", api_status)
        
        info_widget.update(table)
    
    async def load_index_async(self) -> None:
        """Load index asynchronously"""
        progress_label = self.query_one("#progress-label", Static)
        progress_bar = self.query_one("#progress-bar", Static)
        
        def update_progress(status: str, fraction: float):
            self.call_from_thread(progress_label.update, status)
            
            # Create progress bar
            width = 30
            filled = int(width * fraction)
            bar = f"[{'#' * filled}{'-' * (width - filled)}] {int(fraction * 100)}%"
            self.call_from_thread(progress_bar.update, bar)
        
        # Run in separate thread to avoid blocking UI
        def load_index_thread():
            if not INDEX_FILE.exists():
                update_progress("Building new index...", 0.0)
                success = self.rag_system.build_index(update_progress)
            else:
                update_progress("Loading existing index...", 0.0)
                success = self.rag_system.load_index(update_progress)
            
            # Clear progress after completion
            self.call_from_thread(self.update_system_info)
            time.sleep(2)  # Keep message visible for 2 seconds
            self.call_from_thread(progress_label.update, "")
            self.call_from_thread(progress_bar.update, "")
        
        # Start thread
        threading.Thread(target=load_index_thread, daemon=True).start()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """When model selection changes"""
        if event.select.id == "model-selector":
            self.rag_system.llm_model = event.value
            self.update_system_info()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "send-btn":
            self.send_query()
        elif button_id == "build-index-btn":
            self.load_index_async()
        elif button_id == "clear-chat-btn":
            self.action_clear_chat()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """When user submits input with Enter key"""
        if event.input.id == "query-input":
            self.send_query()
    
    def send_query(self) -> None:
        """Send query to RAG system"""
        # Get query
        input_widget = self.query_one("#query-input", Input)
        query = input_widget.value.strip()
        
        if not query:
            return
        
        # Clear input
        input_widget.value = ""
        
        # Display query in chat log
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write(Panel(f"[bold cyan]You:[/] {query}"))
        
        # Process in separate thread to avoid blocking UI
        def process_query():
            # Display thinking message
            self.call_from_thread(chat_log.write, "[italic]Thinking...[/]")
            
            # Get answer
            result = self.rag_system.ask(query)
            
            # Remove thinking message (last line)
            self.call_from_thread(chat_log.write, "")  # Add empty line
            
            # Display answer
            if result["success"]:
                answer_md = Markdown(result["answer"])
                self.call_from_thread(chat_log.write, Panel(answer_md, title="[bold green]Assistant[/]"))
                
                # Display sources
                if result["sources"]:
                    sources_text = Text.from_markup("[bold blue]Sources:[/]\n")
                    seen_urls = set()
                    
                    for i, source in enumerate(result["sources"], 1):
                        meta = source["metadata"]
                        url = meta.get("url", "")
                        
                        if url in seen_urls:
                            continue
                            
                        seen_urls.add(url)
                        title = meta.get("title", "Untitled")
                        sources_text.append(f"{i}. [bold]{title}[/]\n   {url}\n")
                    
                    self.call_from_thread(chat_log.write, Panel(sources_text, title="[bold blue]References[/]"))
            else:
                self.call_from_thread(chat_log.write, Panel(f"[bold red]Error:[/] {result['answer']}"))
                
            # Update system info
            self.call_from_thread(self.update_system_info)
        
        # Start processing thread
        threading.Thread(target=process_query, daemon=True).start()
    
    def action_clear_chat(self) -> None:
        """Clear chat history"""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.clear()
        self.rag_system.chat_history = []
    
    def action_reload_index(self) -> None:
        """Reload index"""
        self.load_index_async()
    
    def action_change_model(self) -> None:
        """Focus model selector"""
        model_selector = self.query_one("#model-selector", Select)
        model_selector.focus()


# === Entry Point ===
if __name__ == "__main__":
    # Create default CSS file if doesn't exist
    css_path = PROJECT_ROOT / "mlx_vet_rag_tui.css"
    if not css_path.exists():
        with open(css_path, "w") as f:
            f.write("""
/* Main layout */
#app-grid {
    layout: grid;
    grid-size: 2;
    grid-columns: 1fr 3fr;
    height: 100%;
}

#left-panel {
    width: 100%;
    height: 100%;
    padding: 1;
    border: solid green;
}

#right-panel {
    width: 100%;
    height: 100%;
    padding: 1;
    border: solid blue;
}

#models-header {
    dock: top;
    text-align: center;
    background: $accent;
    color: $text;
    padding: 1;
    margin-bottom: 1;
}

#model-selector {
    margin-bottom: 1;
}

#system-info-header {
    text-align: center;
    background: $accent-darken-1;
    color: $text;
    padding: 1;
    margin-top: 1;
    margin-bottom: 1;
}

#build-index-btn, #clear-chat-btn {
    margin-top: 1;
    margin-bottom: 1;
}

/* Chat log */
#chat-log {
    height: 1fr;
    border: solid $accent-darken-2;
    padding: 1;
    overflow-y: scroll;
}

/* Input area */
#input-container {
    height: auto;
    margin-top: 1;
    layout: horizontal;
}

#query-input {
    width: 1fr;
    margin-right: 1;
}

/* Utilities */
#progress-label {
    color: $success;
    text-align: center;
    margin-top: 1;
}

#progress-bar {
    color: $success;
    text-align: center;
}

/* Buttons */
Button {
    margin: 1 0;
}

Button:hover {
    background: $accent;
}
""")
    
    # Start the app
    app = VetRAGApp()
    app.run() 