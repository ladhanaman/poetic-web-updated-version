from gtts import gTTS
import io
from typing import Optional

class AudioEngine:
    """
    Handles Text-to-Speech using Google TTS (gTTS).
    Synchronous implementation for maximum stability on Streamlit Cloud.
    """

    def __init__(self, output_dir: str = "assets/audio"):
        # gTTS detects language automatically, but we enforce English
        self.lang = 'en'
        # 'tld' allows us to change the accent. 
        # 'co.uk' gives a British accent (Emily Dickinson style)
        self.tld = 'co.uk' 

    def synthesize(self, text: str, filename: str = None) -> Optional[bytes]:
        """
        Synthesizes text to speech and returns raw bytes.
        """
        print(f"[SYSTEM] Starting gTTS synthesis for: {text[:30]}...")
        
        try:
            if not text.strip():
                print("[ERROR] AudioEngine received empty text.")
                return None

            # Initialize the Google TTS engine
            tts = gTTS(text=text, lang=self.lang, tld=self.tld, slow=False)
            
            # Create an in-memory file buffer (BytesIO)
            # This acts like a file on a hard drive, but lives in RAM.
            buffer = io.BytesIO()
            
            # Write audio data to the buffer
            tts.write_to_fp(buffer)
            
            # Rewind the buffer to the beginning so it can be read
            buffer.seek(0)
            
            # Get the raw bytes
            audio_bytes = buffer.getvalue()
            
            print(f"[SYSTEM] gTTS Complete. Size: {len(audio_bytes)} bytes.")
            return audio_bytes

        except Exception as e:
            print(f"[ERROR] gTTS Failed: {e}")
            return None