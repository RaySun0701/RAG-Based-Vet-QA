#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LM Studio Native API Test
=========================

Simple test script to demonstrate the usage of the LM Studio Python library.
"""

import os
import sys
from pprint import pprint
from lmstudio import Client

def print_section(title):
    """Print a section title with formatting"""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

def main():
    print_section("LM Studio API Test")
    
    # Initialize LM Studio client - will connect to the default location
    print("Initializing LM Studio client...")
    client = Client()
    
    # List downloaded models
    print_section("Downloaded Models")
    try:
        downloaded_models = client.list_downloaded_models()
        for model in downloaded_models:
            print(f"- {model.model_id}")
    except Exception as e:
        print(f"Error listing downloaded models: {e}")
    
    # List loaded models
    print_section("Loaded Models")
    try:
        loaded_models = client.list_loaded_models()
        for model in loaded_models:
            print(f"- {model.model_id}")
    except Exception as e:
        print(f"Error listing loaded models: {e}")
    
    # Fallback to REST API if native API isn't working properly
    print_section("Fallback to REST API")
    import requests
    
    print("Querying LM Studio REST API...")
    try:
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code == 200:
            models = response.json().get("data", [])
            for model in models:
                print(f"- {model['id']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error connecting to REST API: {e}")
    
    # Simple chat completion using REST API
    print_section("Chat Completion Test (REST API)")
    
    # Define messages
    messages = [
        {"role": "system", "content": "You are a helpful veterinary assistant."},
        {"role": "user", "content": "What should I do if my cat isn't eating?"}
    ]
    
    # Generate completion
    print("Generating response...")
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": "qwen3-8b-mlx",
                "messages": messages,
                "temperature": 0.7,
                "stream": False
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\nResponse:")
            print(data["choices"][0]["message"]["content"])
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error generating response: {e}")
    
    # Close the client
    try:
        print("\nClosing connection...")
        client.close()
    except Exception as e:
        print(f"Error closing client: {e}")
    
    print_section("Test Completed")

if __name__ == "__main__":
    main() 