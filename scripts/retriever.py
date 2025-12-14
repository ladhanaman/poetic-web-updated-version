import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "poetic-camera"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Systems
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

print("Connecting to Gemini Embeddings...")
genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text: str) -> List[float]:
    """Generates embedding using Gemini to match your database schema."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query" 
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

def retrieve_poems(query_narrative: str, top_k=3) -> List[Dict[str, Any]]:
    """
    Pure Vector Search. Fast and efficient.
    """
    print(f"\nSearching Pinecone for: '{query_narrative}'")

    vector = get_embedding(query_narrative)
    if not vector:
        return []
    
    try:
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            include_values=True
        )
    except Exception as e:
        print(f"Pinecone Error: {e}")
        return []
    
    if not results['matches']:
        print("No matches found.")
        return []

    found_poems = []
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