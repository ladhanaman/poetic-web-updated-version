import os
import json
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RAGArchitect:
    """
    The 'Brain' of the RAG pipeline.
    Responsibilities:
    1. Re-ranking: Filtering loose vector matches for logical relevance.
    2. Critiquing: Evaluating the quality of generated output (LLM-as-a-Judge).
    """

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key missing. Check .env file.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile" 

    def select_best_candidates(self, vision_narrative: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Takes a list of raw Pinecone matches and returns the Top K most logically relevant ones.
        """
        print(f"[ARCHITECT] Re-ranking {len(candidates)} candidates...")

        candidates_str = ""
        for i, item in enumerate(candidates):
            meta = item.get('metadata', {})
            candidates_str += f"\n[ID: {i}]\nTitle: {meta.get('title', 'Unknown')}\nText: {meta.get('text', '')[:100]}...\n"

        system_prompt = """
        You are a strict Editor for a poetry collection. 
        Your Task: Select the poems that BEST match the specific imagery and mood of the Observed Scene.
        
        Rules:
        1. Ignore generic matches. Look for specific shared nouns or emotions.
        2. Return strictly a JSON object with a list of the best indices.
        
        Output Format:
        { "best_indices": [0, 4, 11] }
        """

        user_prompt = f"""
        OBSERVED SCENE:
        {vision_narrative}

        CANDIDATE POEMS:
        {candidates_str}

        Select the Top {top_k} most relevant indices.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                response_format={"type": "json_object"} 
            )
            
            result = json.loads(response.choices[0].message.content)
            indices = result.get("best_indices", [])
            indices = indices[:top_k]
            
            selected_poems = [candidates[i] for i in indices if i < len(candidates)]
            
            if not selected_poems:
                return candidates[:top_k]

            return selected_poems

        except Exception as e:
            print(f"[ERROR] Re-ranking failed: {e}")
            return candidates[:top_k]

    def evaluate_quality(self, vision_narrative: str, generated_poem: str, poet_name: str) -> Dict[str, Any]:
        """
        The 'Critic'. Judges the generated poem based on the SPECIFIC poet's style.
        """
        
        system_prompt = f"""
        You are a Literary Critic specializing in the works of {poet_name}.
        Analyze the Generated Poem against the Vision Narrative.
        
        Rubric:
        - Relevance (1-5): Does it describe the scene?
        - Style (1-5): Does it sounded like {poet_name}? 
          (e.g., if Dickinson, look for dashes. If Whitman, look for free verse).
        - Hallucination (Boolean): Is it coherent?
        
        Return JSON:
        {{
            "relevance_score": int,
            "style_score": int,
            "is_hallucinated": bool,
            "feedback": "string"
        }}
        """

        user_prompt = f"""
        VISION NARRATIVE:
        {vision_narrative}

        GENERATED POEM:
        {generated_poem}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"[ERROR] Critique failed: {e}")
            return {"relevance_score": 0, "feedback": "Error in critique."}