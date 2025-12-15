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
        # We use the 70B model because 'Reasoning' requires high intelligence.
        # 8B is too small for subtle poetry analysis.
        self.model = "llama-3.3-70b-versatile" 

    def select_best_candidates(self, vision_narrative: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Takes a list of raw Pinecone matches (e.g., 15 items) and returns the Top K
        most logically relevant ones based on the vision narrative.
        """
        print(f"[ARCHITECT] Re-ranking {len(candidates)} candidates for: '{vision_narrative[:30]}...'")

        # 1. Formatting the data for the LLM
        # We create a clean string representation of the candidates.
        candidates_str = ""
        for i, item in enumerate(candidates):
            meta = item.get('metadata', {})
            # We explicitly list the index 'i' so the LLM can reference it.
            candidates_str += f"\n[ID: {i}]\nTitle: {meta.get('title', 'Unknown')}\nText: {meta.get('text', '')[:100]}...\n"

        # 2. The Engineering Prompt
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
            # 3. The Inference
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Low temperature = Logical/Deterministic
                response_format={"type": "json_object"} # FORCE VALID JSON
            )
            
            # 4. Parsing Logic
            result = json.loads(response.choices[0].message.content)
            indices = result.get("best_indices", [])
            
            # Safety check: ensure we didn't get more than requested
            indices = indices[:top_k]
            
            print(f"[ARCHITECT] Selected indices: {indices}")

            # 5. Retrieve the actual objects
            # We map the integer indices back to the original list objects
            selected_poems = [candidates[i] for i in indices if i < len(candidates)]
            
            # Fallback: If LLM picked nothing, return the original top k
            if not selected_poems:
                print("[WARNING] Architect returned 0 matches. Falling back to vector search.")
                return candidates[:top_k]

            return selected_poems

        except Exception as e:
            print(f"[ERROR] Re-ranking failed: {e}")
            # Graceful Degradation: If Re-ranker crashes, just give the first 3 (Pinecone's best guess)
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

if __name__ == "__main__":
    # Unit Test to ensure the Logic works before hooking it to UI
    architect = RAGArchitect()
    
    # Mock Data
    test_narrative = "A dark, stormy ocean with a single lighthouse."
    test_candidates = [
        {"metadata": {"title": "Poem A", "text": "The sun is happy and bright."}},    # Bad Match
        {"metadata": {"title": "Poem B", "text": "The waves crashed against the stone."}}, # Good Match
        {"metadata": {"title": "Poem C", "text": "I like to eat apples."}},            # Bad Match
        {"metadata": {"title": "Poem D", "text": "The light stands alone in the dark."}}   # Good Match
    ]

    # Test Re-ranking
    print("--- Testing Re-ranking ---")
    results = architect.select_best_candidates(test_narrative, test_candidates, top_k=2)
    for res in results:
        print(f"Selected: {res['metadata']['title']}")

    # Test Critique
    print("\n--- Testing Critique ---")
    poem = "The Waves - struck the Stone -\nAnd the Light - watched on -"
    score = architect.evaluate_quality(test_narrative, poem)
    print(score)