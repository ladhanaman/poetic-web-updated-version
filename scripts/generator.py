import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- STYLE CONFIGURATION ---
POET_PROMPTS = {
    "Emily Dickinson": """
        You are the ghost of Emily Dickinson.
        Style: Compressed, enigmatic, and metaphysical.
        Rules:
        1. Use capitalizations for Emphasis (e.g., "The Soul").
        2. Use the Em-Dash (—) frequently for pauses.
        3. Keep it short (4-10 lines).
        4. Focus on the soul, death, nature, and the self.
    """,
    "Percy Bysshe Shelley": """
        You are the ghost of Percy Bysshe Shelley.
        Style: Romantic, revolutionary, and sublime.
        Rules:
        1. Use rich, flowery imagery and complex emotional landscapes.
        2. Focus on the power of nature (wind, mountains, sky) and the spirit of freedom.
        3. Do NOT use Dickinson's dashes. Use standard, elegant punctuation.
        4. Keep it short (4-8 lines).
    """,
    "Walt Whitman": """
        You are the ghost of Walt Whitman.
        Style: Free verse, expansive, and democratic.
        Rules:
        1. Use long, sprawling lines.
        2. Use "cataloging" (listing things).
        3. Celebrate the self, the body, and the connection between all things.
        4. Tone: Robust, declarative, and optimistic.
    """
}

def generate_poem(vision_narrative: str, reference_poems: List[Dict], poet_name: str, temperature: float = 0.6) -> str:
    """
    Dynamically generates a poem based on the selected poet's persona.
    """
    
    # 1. Get the specific system prompt (Default to Dickinson if not found)
    system_instruction = POET_PROMPTS.get(poet_name, POET_PROMPTS["Emily Dickinson"])

    reference_text = ""
    for i, item in enumerate(reference_poems):
        meta = item['metadata']
        text = meta.get('text', '')
        reference_text += f"\n--- Reference {i+1} ---\n{text}\n"

    print(f"Ghost Writer initialized for {poet_name} with {len(reference_poems)} references.")

    # 2. Dynamic System Prompt
    system_prompt = f"""
    {system_instruction}
    
    Your task: observe a scene (described to you) and write a NEW poem about it.
    
    General Rules:
    1. Use the style, meter, and vocabulary of the provided Reference Poems.
    2. Do NOT copy the references. Use them only as a "style transfer" source.
    3. Do not output any intro text. Just the poem.
    """

    user_prompt = f"""
    SCENE OBSERVED:
    {vision_narrative}

    STYLE REFERENCES:
    {reference_text}

    Write the poem now:
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=os.getenv("GENERATOR_MODEL_ID", "llama-3.3-70b-versatile"),
            temperature=temperature,
            max_tokens=300, 
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Generation Failed: {e}")
        return "The camera is blind,\nThe words wont find,\nA path to you."