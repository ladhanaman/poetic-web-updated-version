import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"

INPUT_FILE = "data/dickinson_clean.txt"
OUTPUT_FILE = "data/dickinson_metadata_dense.json"

def get_dense_tags(poem_text):
    """
    Uses the dense prompt strategy with the reliable Llama 3.3 70B model.
    """
    system_prompt = """
    You are a literary scholar analyzing Emily Dickinson. Prioritize deep subtext, hidden metaphors, and emotional arc in your analysis. Your final output MUST be a JSON object conforming strictly to the schema.
    """
    
    user_prompt = f"""
    INSTRUCTIONS:
    1. Analyze the poem deeply to understand its metaphors.
    2. Extract 'Dense Data' based on that complex understanding.

    Return JSON ONLY with this schema:
    {{
        "concrete_nouns": ["list", "of", "5-7", "highly_specific", "physical_objects", "visible_in_imagery"],
        "themes": ["list", "of", "4-6", "complex", "abstract", "concepts"],
        "mood": ["list", "of", "3", "nuanced", "emotional", "adjectives"],
        "analysis_summary": "A single sentence explaining the poem's deeper meaning."
    }}

    POEM:
    {poem_text}
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
            response_format={"type": "json_object"} 
        )
        return json.loads(completion.choices[0].message.content)
        
    except Exception as e:
        print(f"Error extracting tags: {e}")
        return None

def load_existing_data():
    """Loads progress so we can resume if crashed."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except json.JSONDecodeError:
            print("JSON file corrupted or empty. Starting fresh.")
            return []
    return []

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Could not find {INPUT_FILE}. Run ingestion script first.")
        return

    # 1. Load Raw Poems
    print("Loading poems...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = f.read()
        all_poems = raw_data.split("\n---POEM_SEPARATOR---\n")

    # 2. Check Progress
    processed_data = load_existing_data()
    start_index = len(processed_data)
    
    print(f"Found {start_index} poems already processed.")
    print(f"Starting Deep Analysis on remaining {len(all_poems) - start_index} poems using {MODEL_NAME}...")

    # 3. Processing Loop
    for i in range(start_index, len(all_poems)):
        poem = all_poems[i].strip()
        
        # --- PROSE FILTER ---
        lines = poem.split('\n')
        if len(lines) > 0:
            avg_line_len = sum(len(line) for line in lines) / len(lines)
        else:
            avg_line_len = 0

        # Filter Condition
        if len(poem) < 10 or avg_line_len > 65:
            print(f"  Skipping Poem #{i+1} (Prose/Note detected)")
            
            entry = {"id": f"poem_{i:04d}", "status": "skipped"}
            processed_data.append(entry)
            
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2)
            continue

        print(f"Thinking about Poem #{i+1}...")
        
        tags = get_dense_tags(poem)
        
        if tags:
            entry = {
                "id": f"poem_{i:04d}",
                "text": poem,
                "metadata": tags 
            }
            processed_data.append(entry)
            
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2)
            
  
            time.sleep(0.5)
        else:
            print("Failed to tag. Saving progress and stopping.")
            break

    print(f"Script finished. Total database size: {len(processed_data)}")

if __name__ == "__main__":
    main()