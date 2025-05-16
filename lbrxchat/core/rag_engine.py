#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLX RAG System - Wykorzystuje LM Studio API z modelami MLX
"""

import os
import json
from pathlib import Path
import time
from tqdm import tqdm
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Konfiguracja klienta OpenAI do połączenia z lokalnym LM Studio
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="sk-no-key-required"  # LM Studio nie wymaga konkretnego klucza API
)

# === Modele ===
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"  # Model embeddingów 
LLM_MODEL = "qwen3-8b-mlx"  # Mniejszy, ale wciąż zajebisty model Qwen3 MLX

# === Ścieżki ===
PROJECT_ROOT = Path(__file__).parent.absolute()
CORPUS_PATH = PROJECT_ROOT / "vet_corpus" / "chunked_merck.jsonl"
INDEX_PATH = PROJECT_ROOT / "vet_index_mlx"
INDEX_FILE = INDEX_PATH / "index.json"
INDEX_PATH.mkdir(exist_ok=True, parents=True)

# === Wczytanie korpusu ===
def load_corpus():
    """Wczytuje korpus z plików JSONL"""
    documents = []
    
    print(f"Wczytuję dane z: {CORPUS_PATH}")
    if not CORPUS_PATH.exists():
        print(f"Plik {CORPUS_PATH} nie istnieje. Uruchom najpierw skrypty przygotowujące dane.")
        return []
    
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Wczytywanie dokumentów"):
            doc = json.loads(line)
            documents.append(doc)
    
    print(f"Wczytano {len(documents)} dokumentów.")
    return documents

# === Tworzenie embeddingów ===
def create_embeddings(texts):
    """Generuje embeddingi dla listy tekstów przy użyciu modelu z LM Studio"""
    batch_size = 8  # Mniejszy batch dla stabilności
    all_embeddings = []
    
    # Przetwarzanie w batchach
    for i in tqdm(range(0, len(texts), batch_size), desc="Generowanie embeddingów"):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.2)  # Zwiększone opóźnienie dla stabilności
        except Exception as e:
            print(f"Błąd przy generowaniu embeddingów: {e}")
            # Próba ponowienia dla mniejszego batcha
            if len(batch) > 1:
                print("Ponawianie z mniejszym batchem...")
                for text in batch:
                    try:
                        response = client.embeddings.create(
                            input=[text],
                            model=EMBEDDING_MODEL
                        )
                        all_embeddings.append(response.data[0].embedding)
                        time.sleep(0.3)
                    except Exception as e2:
                        print(f"Pominięto tekst z powodu błędu: {e2}")
                        all_embeddings.append([0.0] * 768)  # Placeholder embedding
    
    return all_embeddings

# === Budowanie indeksu ===
def build_index(documents):
    """Buduje indeks wektorowy dla dokumentów"""
    print("Ekstrakcja treści dokumentów...")
    texts = [doc["content"] for doc in documents]
    
    print(f"Generowanie embeddingów dla {len(texts)} dokumentów...")
    embeddings = create_embeddings(texts)
    
    # Zapisywanie indeksu
    print("Zapisywanie indeksu...")
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
    
    print(f"Indeks zapisany do: {INDEX_FILE}")
    return index_data

# === Wyszukiwanie w indeksie ===
def vector_search(query, index_data, top_k=5):
    """Wyszukuje najbardziej podobne dokumenty do zapytania"""
    # Generowanie embeddingu dla zapytania
    query_embedding_resp = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    query_embedding = query_embedding_resp.data[0].embedding
    
    # Przygotowanie embeddingów z dokumentów
    doc_embeddings = np.array([doc["embedding"] for doc in index_data])
    
    # Obliczenie podobieństwa
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # Zwróć top k najbardziej podobnych dokumentów
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    
    for idx in top_indices:
        results.append({
            "content": index_data[idx]["content"],
            "metadata": index_data[idx]["metadata"],
            "score": float(similarities[idx])
        })
    
    return results

# === Generowanie odpowiedzi ===
def generate_answer(query, context, chat_history=None):
    """Generuje odpowiedź na pytanie w oparciu o kontekst"""
    if chat_history is None:
        chat_history = []
    
    # Przygotowanie kontekstu
    context_text = "\n\n".join([doc["content"] for doc in context])
    
    # Przygotowanie systemu i wiadomości
    messages = [
        {"role": "system", "content": f"Jesteś asystentem weterynaryjnym. Odpowiadasz na pytania w oparciu o dostarczone źródła informacji. Jeśli nie znasz odpowiedzi, przyznaj się. Informacje źródłowe:\n\n{context_text}"},
        *chat_history,
        {"role": "user", "content": query}
    ]
    
    # Wywołanie API
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,  # Lekko losowe odpowiedzi dla naturalności
            max_tokens=1024
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Błąd podczas generowania odpowiedzi: {e}")
        return "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."

# === Przygotowanie chatbota ===
def run_rag_chatbot():
    """Uruchamia chatbota RAG w konsoli"""
    print("\nWeterynaryjny Asystent RAG bazujący na MLX")
    print("==========================================")
    print(f"Używany model LLM: {LLM_MODEL}")
    print(f"Używany model embeddingów: {EMBEDDING_MODEL}")
    
    # Sprawdzenie czy indeks istnieje, jeśli nie, budowanie indeksu
    if not INDEX_FILE.exists():
        print("Indeks nie istnieje. Tworzenie indeksu...")
        documents = load_corpus()
        if not documents:
            return
        index_data = build_index(documents)
    else:
        print("Wczytywanie istniejącego indeksu...")
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        print(f"Wczytano indeks z {len(index_data)} dokumentami.")
    
    # Pamięć chatbota (historia konwersacji)
    chat_history = []
    
    print("\nMożesz zadawać pytania weterynaryjne. Wpisz 'quit' lub 'exit' aby zakończyć.")
    
    while True:
        query = input("\n> ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        # Wyszukanie odpowiednich kontekstów
        print("Wyszukiwanie odpowiednich informacji...")
        search_results = vector_search(query, index_data, top_k=5)
        
        # Generowanie odpowiedzi
        print("Generowanie odpowiedzi...")
        answer = generate_answer(query, search_results, chat_history)
        print("\nOdpowiedź:")
        print(answer)
        
        # Wyświetlenie źródeł
        print("\nŹródła:")
        seen_urls = set()
        for i, result in enumerate(search_results, 1):
            meta = result["metadata"]
            url = meta.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title = meta.get("title", "Bez tytułu")
            print(f"[{i}] {title}\n    {url}")
        
        # Aktualizacja historii
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
        
        # Zachowaj tylko ostatnie 10 wiadomości w historii (5 wymian)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

if __name__ == "__main__":
    run_rag_chatbot() 