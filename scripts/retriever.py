import os
from typing import List, Dict, Any
from pinecone import Pinecone

from scripts.config import get_setting
from scripts.openai_client import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
    get_openai_client,
)

# --- Configuration ---
PINECONE_API_KEY = get_setting("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is not configured. Add it to the environment, .env, "
        "or Streamlit secrets before starting retrieval."
    )

# Best Practice: Fallback to v2 if .env is missing, but prefer env var
PINECONE_INDEX_NAME = get_setting("PINECONE_INDEX_NAME", "poetic-camera-v2")

# Initialize Systems
print(f"Connecting to Pinecone Index: {PINECONE_INDEX_NAME}...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print("Connecting to OpenAI Embeddings...")
print(f"Using OpenAI embedding model: {OPENAI_EMBEDDING_MODEL}")

def get_embedding(text: str) -> List[float]:
    """Generate a query embedding compatible with the Pinecone index."""
    try:
        response = get_openai_client().embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
            dimensions=OPENAI_EMBEDDING_DIMENSIONS,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

def retrieve_poems(query_narrative: str, top_k=3, namespace=None) -> List[Dict[str, Any]]:
    """
    Pure Vector Search. Fast and efficient.
    """
    print(f"\nSearching Pinecone '{PINECONE_INDEX_NAME}' for: '{query_narrative}' (Namespace: {namespace})")

    vector = get_embedding(query_narrative)
    if not vector:
        return []
    
    try:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False, # OPTIMIZATION: Set to False to save bandwidth
            namespace=namespace
        )
    except Exception as e:
        print(f"Pinecone Error: {e}")
        return []
    
    if not results['matches']:
        print("No matches found.")
        return []

    found_poems = []
    print("\nTop chunckes using bi-encoder cosine similarity search:")
    print(f"Found {len(results['matches'])} matches.")
    
    for match in results['matches']:
        found_poems.append(match)
        title = match['metadata'].get('title', 'Unknown')
        score = match['score']
        print(f"   ★ {title} (Similarity: {score:.4f})")
        
    return found_poems

if __name__ == "__main__":
    # Test
    retrieve_poems("A Serene poem about Nature and Solitude.")
