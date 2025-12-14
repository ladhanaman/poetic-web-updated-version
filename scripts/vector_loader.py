import json
import os
import argparse
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

load_dotenv()

# --- CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") # Picked up from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API Keys! Check your .env file.")

# Initialize Clients
print(f"Connecting to Pinecone Index: {PINECONE_INDEX_NAME}...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print("Connecting to Gemini...")
genai.configure(api_key=GEMINI_API_KEY)

def build_semantic_string(poem_obj: Dict[str, Any]) -> str:
    """
    Converts metadata into a search-optimized string.
    """
    meta = poem_obj.get("metadata", {})
    noun_str = ", ".join(meta.get("concrete_nouns", []))
    theme_str = ", ".join(meta.get("themes", []))
    mood_str = ", ".join(meta.get("mood", []))
    
    # We construct a sentence that mimics a user's potential query
    narrative = f"A {mood_str} poem about {theme_str}, featuring imagery of {noun_str}."
    return narrative

def load_data(json_file: str, namespace: str):
    print(f"\n--- INGESTION PROTOCOL STARTED ---")
    print(f"Target Namespace: '{namespace}'")
    print(f"Source File:      {json_file}")
    
    if not os.path.exists(json_file):
        print(f"[ERROR] File not found: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        poems = json.load(f)
        
    # Filter out skipped items (from the prose filter)
    valid_poems = [p for p in poems if p.get("status") != "skipped"]
    print(f"Loaded {len(valid_poems)} valid poems (out of {len(poems)} total).")

    batch_size = 50
    vectors_to_upsert = []

    print("Starting Embedding & Upload Loop...")

    for i, poem in enumerate(valid_poems):
        semantic_text = build_semantic_string(poem)
        
        # 1. Generate Embedding
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=semantic_text,
                task_type="retrieval_document" 
            )
            embedding = response['embedding']
        except Exception as e:
            print(f" [!] Embedding failed for ID {poem.get('id')}: {e}")
            continue

        # 2. Build Payload
        vector_payload = {
            "id": poem.get("id"), 
            "values": embedding,
            "metadata": {
                "text": poem.get("text"),
                "title": f"Poem {poem.get('id')}",
                "semantic_string": semantic_text,
                "author": namespace # Tagging the author is crucial for multi-tenancy
            }
        }
        vectors_to_upsert.append(vector_payload)

        # 3. Upsert Batch
        if len(vectors_to_upsert) >= batch_size:
            try:
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                print(f"   ✓ Uploaded batch {i+1}/{len(valid_poems)}")
                vectors_to_upsert = [] # Reset
                time.sleep(0.5) # Rate limit safety
            except Exception as e:
                 print(f"   [!] Pinecone Upload Error: {e}")

    # 4. Final Batch
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        print(f"   ✓ Uploaded final batch.")

    print(f"--- SUCCESS: Namespace '{namespace}' is ready. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the JSON metadata file")
    parser.add_argument("--namespace", required=True, help="Target Pinecone namespace (e.g. 'dickinson', 'poe')")
    args = parser.parse_args()
    
    load_data(args.file, args.namespace)