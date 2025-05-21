#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LBRXCHAT RAG System
===================

Retrieval-Augmented Generation (RAG) implementation for LBRXCHAT.
"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Local imports
from lbrxchat.core.models import MLXModelManager
from lbrxchat.core.config import (
    CORPUS_PATH, INDEX_PATH, INDEX_FILE, DEFAULT_LLM_MODEL, 
    DEFAULT_EMBEDDING_MODEL, SYSTEM_PROMPTS, LMSTUDIO_HOST,
    DEFAULT_TOP_K, EMBEDDING_DIMENSION, CHAT_TEMPERATURE
)


class VetRAGSystem:
    """Core RAG system for question answering"""
    
    def __init__(self, model_manager: MLXModelManager):
        self.model_manager = model_manager
        self.index_data = None
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        self.llm_model = DEFAULT_LLM_MODEL
        self.chat_history = []
        
        # Ensure index directory exists
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
    
    def load_corpus(self, callback: Optional[Callable[[str, float], None]] = None) -> List[Dict[str, Any]]:
        """Load corpus of documents"""
        documents = []
        
        if not CORPUS_PATH.exists():
            if callback:
                callback("Corpus file not found", 1.0)
            return []
        
        try:
            total_lines = sum(1 for _ in open(CORPUS_PATH, 'r', encoding='utf-8'))
            
            with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    doc = json.loads(line)
                    documents.append(doc)
                    if callback:
                        callback(f"Loading documents: {i+1}/{total_lines}", (i+1)/total_lines)
        except Exception as e:
            if callback:
                callback(f"Error loading corpus: {e}", 1.0)
            return []
        
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
        
        # Extract text content from documents
        texts = []
        for doc in documents:
            # Handle different document formats
            if "content" in doc:
                texts.append(doc["content"])
            elif "text" in doc:
                texts.append(doc["text"])
            else:
                # Try to find any field with text content
                text_candidates = [v for k, v in doc.items() 
                                  if isinstance(v, str) and len(v) > 50]
                if text_candidates:
                    texts.append(text_candidates[0])
                else:
                    # Skip document if no suitable text found
                    continue
        
        # For this implementation, we'll simulate embeddings with random vectors
        # In a real implementation, you would use an embedding model here
        if callback:
            callback("Generating embeddings (simulated)...", 0.2)
        
        # Generate embeddings (random for demo)
        embeddings = [np.random.rand(EMBEDDING_DIMENSION).tolist() for _ in range(len(texts))]
        
        # Save embeddings to index
        if callback:
            callback("Saving index...", 0.8)
        
        index_data = []
        for i, (doc, embedding, text) in enumerate(zip(documents, embeddings, texts)):
            metadata = {}
            
            # Extract metadata from document
            if "metadata" in doc:
                metadata = doc["metadata"]
            else:
                # Try to extract common metadata fields
                for field in ["title", "url", "source", "author", "date", "id"]:
                    if field in doc:
                        metadata[field] = doc[field]
            
            # Ensure minimum metadata
            if "title" not in metadata:
                metadata["title"] = f"Document {i+1}"
            
            if "id" not in metadata:
                metadata["id"] = str(i)
            
            index_data.append({
                "id": i,
                "content": text,
                "metadata": metadata,
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
    
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
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
        context_text = "\n\n".join([f"Document {i+1}:\n{doc['content']}" 
                                   for i, doc in enumerate(contexts)])
        
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
                    temperature=CHAT_TEMPERATURE,
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
                f"{LMSTUDIO_HOST}/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": CHAT_TEMPERATURE,
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
    
    def ask(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """Ask a question and get answer with sources"""
        # Search for relevant contexts
        contexts = self.search(query, top_k=top_k)
        
        # Generate answer
        result = self.generate_answer(query, contexts)
        
        return result 