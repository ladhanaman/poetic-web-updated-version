import json
import os
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from pinecone import Pinecone 
import google.generativeai as genai


load_dotenv() 


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "poetic-camera"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API Keys! Did you set up your .env file?")

# Initialize Clients
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print("Connecting to Gemini...")
genai.configure(api_key=GEMINI_API_KEY)
print("Systems Online.")

def build_semantic_string(poem_obj: Dict[str, Any]) -> str:
    """
    Converts raw poem JSON into a targeted semantic string.
    Focuses ONLY on Mood, Themes, and Concrete Nouns.
    """
    meta = poem_obj.get("metadata", {})
    
    noun_str = ", ".join(meta.get("concrete_nouns", []))
    theme_str = ", ".join(meta.get("themes", []))
    mood_str = ", ".join(meta.get("mood", []))
    
    narrative = f"A {mood_str} poem about {theme_str}, featuring imagery of {noun_str}."
        
    return narrative


def load_data():
    try:
        with open("data/dickinson_metadata_dense.json", "r") as f:
            poems = json.load(f)
        print(f"Loaded {len(poems)} poems from JSON.")
    except FileNotFoundError:
        print("Error: 'dickinson_metadata_dense.json' not found!")
        return

    batch_size = 50
    vectors_to_upsert = []
    
    print("Starting Batch Processing...")

    for i, poem in enumerate(poems):
        # Create the Semantic String
        semantic_text = build_semantic_string(poem)
        
        # B. Generate Embedding (The "Translation" to Math)
        # We use 'task_type="retrieval_document"' because this data is being stored for searching
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=semantic_text,
                task_type="retrieval_document" 
            )
            embedding = response['embedding']
        except Exception as e:
            print(f"Error embedding poem {poem.get('id', 'unknown')}: {e}")
            continue

        # Prepare Pinecone Payload
        # We store the 'text' in metadata so we can print it later without querying a separate DB
        vector_payload = {
            "id": poem.get("id"), 
            "values": embedding,
            "metadata": {
                "text": poem.get("text"),
                "title": f"Poem {poem.get('id')}",
                "semantic_string": semantic_text # Good for debugging later
            }
        }
        vectors_to_upsert.append(vector_payload)

        # D. Upsert when Batch is Full
        if len(vectors_to_upsert) >= batch_size:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Upserted batch {i+1}/{len(poems)}")
            vectors_to_upsert = [] # Reset batch

    # Final Upsert (for any leftovers)
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        print(f"Upserted final batch. Total processed.")

if __name__ == "__main__":
    load_data()