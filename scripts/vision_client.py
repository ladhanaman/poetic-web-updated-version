import os
import json
import base64
import io
from typing import BinaryIO, Union
from dotenv import load_dotenv
from PIL import Image

from scripts.openai_client import (
    OPENAI_VISION_MODEL,
    get_openai_client,
    sampling_parameters,
)

load_dotenv()
VISION_MODEL_ID = OPENAI_VISION_MODEL

ImageSource = Union[str, os.PathLike[str], BinaryIO]


def analyze_image(image_file: ImageSource | None) -> str:
    """
    Takes a Streamlit UploadedFile (or file-like object), 
    processes it, and analyzes it using OpenAI Vision.
    """
    
    if image_file is None:
        return ""

    print(f"[SYSTEM] Analyzing image with model: {VISION_MODEL_ID}...")
    
    try:
        # 1. Load and Resize
        with Image.open(image_file) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to optimize token usage and latency
            img.thumbnail((512, 512))
            
            # 2. Convert to Base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 3. The Prompt
        system_prompt = "You are a poetic assistant. Return ONLY valid JSON."
        user_prompt = """
        Analyze this image for a 19th-century poem. Identify:
        1. Mood: Single adjective.
        2. Themes: 2-4 abstract concepts.
        3. Concrete Nouns: 3-6 physical objects.

        Return strictly this JSON:
        {
            "mood": "str",
            "themes": ["str", "str"],
            "concrete_nouns": ["str", "str"]
        }
        """

        # 4. API Call
        response = get_openai_client().chat.completions.create(
            model=VISION_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            **sampling_parameters(VISION_MODEL_ID, max_completion_tokens=200),
        )
        
        # 5. Parse
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned an empty vision response.")

        result = json.loads(content)
        
        # 6. Construct String
        noun_str = ", ".join(result.get('concrete_nouns', []))
        theme_str = ", ".join(result.get('themes', []))
        mood_str = result.get('mood', 'Unknown')
        
        narrative = f"A {mood_str} poem about {theme_str}, featuring imagery of {noun_str}."
        
        print(f"[SUCCESS] Generated Query: '{narrative}'")
        return narrative

    except Exception as e:
        print(f"[ERROR] Vision Bridge Failed: {e}")
        # Fallback suggestion for the logs
        if "model_decommissioned" in str(e):
            print("CRITICAL: The model ID is invalid. Check the OpenAI model access for the latest model.")
        return f"ERROR: {str(e)}"

if __name__ == "__main__":
    if os.path.exists("test.jpg"):
        with open("test.jpg", "rb") as f:
            print(analyze_image(f))
    else:
        print("No test.jpg found.")
