import os
import cohere
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RAGArchitect:
    """
    The 'Discriminator' of the RAG pipeline.
    
    Now powered by Cohere Rerank 3.5 (API).
    It replaces the slow LLM-based selection with a fast, mathematical reranking model.
    """

    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            print("[WARNING] COHERE_API_KEY missing. Reranker will fail.")
            
        self.client = cohere.Client(self.api_key)
        self.model = "rerank-english-v3.0" 

    def select_best_candidates(self, vision_narrative: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Uses Cohere's semantic reranker to sort loose Pinecone matches by true relevance.
        
        Args:
            vision_narrative: The query (what the camera saw).
            candidates: List of Pinecone results (dicts with 'metadata').
            top_k: Number of final poems to return.
            
        Returns:
            The top_k strictly most relevant document objects.
        """
        if not candidates:
            return []
            
        print(f"\nReranking {len(candidates)} candidates via Cohere reranker 3.5->\nTop 3: ")

        # 1. Prepare Documents for the API
        # Cohere expects a list of strings. We extract the text content.
        docs_text = [doc.get('metadata', {}).get('text', '') for doc in candidates]

        try:
            # 2. Call the API
            response = self.client.rerank(
                model=self.model,
                query=vision_narrative,
                documents=docs_text,
                top_n=top_k
            )
            
            # 3. Map Results back to Original Objects
            # Cohere returns indices (e.g., "Document 4 is #1"). 
            # We use these indices to grab the original full dictionary from 'candidates'.
            ranked_candidates = []
            
            for result in response.results:
                original_doc = candidates[result.index]
                print(f"[{original_doc.get('metadata', {}).get('title', '').lower().replace('poem poem', 'Poem')} : {result.index}]")
                ranked_candidates.append(original_doc)

            return ranked_candidates

        except Exception as e:
            print(f"[ERROR] Cohere Rerank failed: {e}")
            return candidates[:top_k]
