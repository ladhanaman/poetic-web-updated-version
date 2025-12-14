import os
import json
import time
import argparse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

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

def load_existing_data(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return json.loads(f.read().strip() or "[]")
        except:
            return []
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cleaned .txt file")
    parser.add_argument("--output", required=True, help="Output .json file")
    parser.add_argument("--loose", action="store_true", help="Disable strict line-length checks (for Whitman)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return

    print("Loading poems...")
    with open(args.input, "r", encoding="utf-8") as f:
        all_poems = f.read().split("\n---POEM_SEPARATOR---\n")

    processed_data = load_existing_data(args.output)
    start_index = len(processed_data)
    
    print(f"Resuming from index {start_index}...")

    for i in range(start_index, len(all_poems)):
        poem = all_poems[i].strip()
        
        # --- LOGIC UPDATE FOR WHITMAN ---
        lines = poem.split('\n')
        avg_line_len = sum(len(l) for l in lines) / len(lines) if lines else 0
        
        # If --loose is set (Whitman), we allow longer lines (up to 200 chars)
        max_len = 200 if args.loose else 65 
        
        if len(poem) < 10 or avg_line_len > max_len:
            print(f"  Skipping Poem #{i+1} (Filter: {avg_line_len:.1f} chars/line)")
            processed_data.append({"id": f"poem_{i:04d}", "status": "skipped"})
            # Save placeholder to maintain index alignment
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2)
            continue

        print(f"Tagging Poem #{i+1}...")
        tags = get_dense_tags(poem)
        
        if tags:
            processed_data.append({
                "id": f"poem_{i:04d}",
                "text": poem,
                "metadata": tags 
            })
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2)
            time.sleep(0.5)
        else:
            print("Failed. Saving and stopping.")
            break

if __name__ == "__main__":
    main()