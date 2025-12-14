import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_poem(vision_narrative: str, reference_poems: List[Dict],temperature: float = 0.6) -> str:
    """
    Inputs:
        vision_narrative: The description of what the camera saw.
        reference_poems: The Top 3 poems retrieved from the DB.
    Output:
        A new, original poem in the style of Emily Dickinson.
    """
    

    reference_text = ""
    for i, item in enumerate(reference_poems):
        meta = item['metadata']
        text = meta.get('text', '')

        reference_text += f"\n--- Reference {i+1} ---\n{text}\n"

    print(f"Ghost Writer initialized with {len(reference_poems)} references.")

    #System Prompt
    system_prompt = """
    You are the ghost of Emily Dickinson. 
    You do not speak like a modern assistant. You speak only in poetry.
    
    Your task: observe a scene (described to you) and write a NEW poem about it.
    
    Rules:
    1. Use the style, meter, and vocabulary of the provided Reference Poems.
    2. Do NOT copy the references. Use them only as a "style transfer" source.
    3. Keep it short (4-12 lines).
    4. Use capitalization for Emphasis.
    5. Use the Em-Dash (—) for pauses, NOT the hyphen (-). This is crucial.
    6. Do not output any intro text (like "Here is a poem"). Just the poem.
    """

    #The User Prompt
    user_prompt = f"""
    SCENE OBSERVED:
    {vision_narrative}

    STYLE REFERENCES:
    {reference_text}

    Write the poem now:
    """

    #The Generation
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=200,
        )
        
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Generation Failed: {e}")
        return "The camera is blind,\nThe words wont find,\nA path to you."

if __name__ == "__main__":
    #Test
    test_narrative = "A Serene poem about Nature and Solitude."
    test_refs = [
        {"metadata": {"text": "I'm Nobody! Who are you?\nAre you - Nobody - too?"}},
        {"metadata": {"text": "Hope is the thing with feathers -\nThat perches in the soul -"}}
    ]
    
    print(generate_poem(test_narrative, test_refs))