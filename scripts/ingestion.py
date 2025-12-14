import sys
import argparse
from pathlib import Path


DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "dickinson_complete.txt"
OUTPUT_FILE = DATA_DIR / "dickinson_clean.txt"

def load_text(path: Path) -> str:

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_and_split(raw_text: str) -> list[str]:

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start = raw_text.find(start_marker)
    end = raw_text.find(end_marker)
    
    if start != -1 and end != -1:
        content = raw_text[start:end].split("\n", 1)[1]
    else:
        content = raw_text

    # Normalize Roman Numerals & Headers
    # We want to remove lines that are JUST Roman numerals (I., XIV.) or Category titles (LIFE, LOVE)
    # This Regex looks for lines that are just uppercase words or Roman numerals
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines for now, we'll handle poem breaks later
        if not line:
            cleaned_lines.append("")
            continue
            
        # Filter out metadata lines like "IV.", "PART ONE: LIFE", "Written in 1862"
        # If line is short and uppercase/roman, skip it
        if len(line) < 20 and line.isupper():
            continue
        
        cleaned_lines.append(line)

    # Rejoin to process as blocks
    text_block = "\n".join(cleaned_lines)
    
    # Split by Double Newline (The standard poem delimiter)
    raw_chunks = text_block.split("\n\n\n") # Gutenberg uses roughly 3 newlines between poems
    
    valid_poems = []
    for chunk in raw_chunks:
        clean_chunk = chunk.strip()
        # A valid Dickinson poem is usually at least 30 chars
        if len(clean_chunk) > 30:
            valid_poems.append(clean_chunk)
            
    return valid_poems

def main():
    parser = argparse.ArgumentParser(description="Clean raw Project Gutenberg text files.")
    parser.add_argument("--input", required=True, help="Path to raw input .txt file")
    parser.add_argument("--output", required=True, help="Path to save cleaned .txt file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading: {input_path}")
    try:
        raw_text = load_text(input_path)
    except Exception as e:
        print(e)
        return

    print("Processing text...")
    poems = clean_and_split(raw_text)
    
    print(f"Extracted {len(poems)} poems.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n---POEM_SEPARATOR---\n".join(poems))
        
    print(f"Saved clean dataset to {output_path}")

if __name__ == "__main__":
    main()