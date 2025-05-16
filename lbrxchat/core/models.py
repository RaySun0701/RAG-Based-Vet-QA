#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LBRXCHAT Model Management
=========================

Model management system for LBRXCHAT - handles loading/unloading of LLM models,
JIT management with TTL, and LM Studio integration.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# LM Studio integration
import requests
try:
    from lmstudio import Client as LMStudioClient
    LMSTUDIO_NATIVE_API = True
except ImportError:
    LMSTUDIO_NATIVE_API = False

# Local imports
from lbrxchat.core.config import (
    MLX_MODELS_PATH, MODEL_TTL, DEFAULT_LLM_MODEL,
    LMSTUDIO_HOST, LMSTUDIO_API_KEY
)


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
                response = requests.get(f"{LMSTUDIO_HOST}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["id"] for model in data.get("data", [])]
            except Exception as e:
                print(f"Error fetching models via native API: {e}")
        
        # Fallback to REST API
        if not models:
            try:
                response = requests.get(f"{LMSTUDIO_HOST}/v1/models")
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
            
            # For this implementation, we'll use the existing LM Studio server on port 1234
            # In a real implementation, we would start mlx-serve here
            port = int(LMSTUDIO_HOST.split(":")[-1]) if ":" in LMSTUDIO_HOST else 1234
            
            self.active_servers[model_name] = {
                "port": port,
                "last_used": datetime.now(),
                "ttl_minutes": ttl_minutes,
                "process": None
            }
            
            return port
    
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


# Registry of custom models
_MODEL_REGISTRY = {}


def register_model(name: str, model_type: str, context_length: int, system_prompt: str) -> None:
    """Register a custom model in the system"""
    _MODEL_REGISTRY[name] = {
        "name": name,
        "type": model_type,
        "context_length": context_length,
        "system_prompt": system_prompt
    }


def get_model_info(name: str) -> Optional[Dict[str, Any]]:
    """Get information about a registered model"""
    return _MODEL_REGISTRY.get(name)


def list_registered_models() -> List[str]:
    """Get list of all registered models"""
    return list(_MODEL_REGISTRY.keys()) 