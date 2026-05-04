import os
from gtts import gTTS
import pygame
import time

def text_to_speech(text, lang='en'):
    """
    Converts text to spoken audio using Google TTS and plays it through the speakers.
    
    Flow:
    1. gTTS converts the text string into an MP3 audio file
    2. pygame loads and plays that MP3 file through the speakers
    3. After playback, the temporary file is deleted
    
    Args:
        text: The text string to speak aloud (e.g., the LLM summary)
        lang: Language code (default: 'en' for English)
    """
    # Step 1: Convert text to an MP3 audio file using Google TTS
    print("\n[Generating speech audio...]")
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Step 2: Save the generated audio to a temporary file
    audio_file = "temp_response.mp3"
    tts.save(audio_file)
    
    # Step 3: Initialize pygame's audio mixer and play the file
    print("[Speaking...]\n")
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    
    # Step 4: Wait until the audio finishes playing
    # tick(10) means check 10 times per second whether audio is still playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    # Step 5: Clean up - close pygame and delete the temporary file
    pygame.mixer.music.unload()
    pygame.mixer.quit()
    
    try:
        os.remove(audio_file)
    except Exception:
        pass
    
    print("[Speech complete.]")


if __name__ == "__main__":
    # ─── Test: Run the TTS module standalone ───
    print("=== Text-to-Speech Test ===\n")
    test_message = "Hello! I am your AI News Voice Assistant. Today's top story: AI is transforming industries worldwide."
    print(f'Speaking: "{test_message}"')
    text_to_speech(test_message)
