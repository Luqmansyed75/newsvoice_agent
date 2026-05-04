import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import os

# ─── Step 1: Record audio from the microphone ───
def record_audio(duration=5, sample_rate=16000):
    """
    Records audio from the user's microphone for a given duration.
    
    Args:
        duration: How many seconds to record (default: 5 seconds)
        sample_rate: Audio quality setting (16000 Hz is what Whisper expects)
    
    Returns:
        audio_data: Raw audio as a NumPy array
        sample_rate: The sample rate used
    """
    print(f"\n[Listening for {duration} seconds... Speak now!]")
    
    # Record audio from the default microphone
    audio_data = sd.rec(
        int(duration * sample_rate),  # Total number of audio samples to capture
        samplerate=sample_rate,        # 16kHz = standard for speech recognition
        channels=1,                    # Mono audio (single channel, not stereo)
        dtype='float32'                # Data type for the audio samples
    )
    
    # Block execution until the recording is complete
    sd.wait()
    
    print("[Recording complete!]")
    return audio_data, sample_rate

# ─── Step 2: Save the recorded audio to a temporary WAV file ───
def save_audio(audio_data, sample_rate, filename="temp_recording.wav"):
    """
    Saves the recorded NumPy audio array to a .wav file on disk.
    Whisper requires a file path as input, so we save it temporarily.
    """
    # Flatten to 1D array in case it has extra dimensions
    audio_flat = audio_data.flatten()
    
    # Convert float32 audio (-1.0 to 1.0) to int16 format (-32768 to 32767)
    # WAV files expect integer samples, not floating point
    audio_int16 = np.int16(audio_flat * 32767)
    
    # Write the audio data to a WAV file
    wav.write(filename, sample_rate, audio_int16)
    return filename

# ─── Step 3: Transcribe the audio directly using Whisper ───
def transcribe_audio(audio_array, model_size="base"):
    """
    Uses OpenAI Whisper (running locally) to convert the raw audio array to text.
    By passing the numpy array directly, we bypass the need to install FFmpeg!
    
    Args:
        audio_array: 1D NumPy array of audio data (float32)
        model_size: Whisper model size
    
    Returns:
        text: The transcribed text string
    """
    print(f"[Loading Whisper '{model_size}' model...]")
    
    import warnings
    # Ignore the FP16 warning on CPU
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    
    # Load the Whisper model
    model = whisper.load_model(model_size)
    
    print("[Transcribing your speech...]")
    
    # Run the transcription directly on the raw audio array
    result = model.transcribe(audio_array)
    
    # Extract just the text
    text = result["text"].strip()
    
    print(f'[You said: "{text}"]')
    return text

# ─── Main Function: Full Speech-to-Text Pipeline ───
def speech_to_text(duration=5, model_size="base"):
    """
    Complete STT pipeline:
    1. Record audio from the microphone
    2. Transcribe it directly using Whisper
    3. Return the transcribed text as `query`
    """
    # Step 1: Record from microphone
    audio_data, sample_rate = record_audio(duration=duration)
    
    # Step 2: Flatten the audio to a 1D float32 array (which Whisper expects)
    audio_flat = audio_data.flatten()
    
    # Step 3: Transcribe using Whisper (passing array directly skips FFmpeg!)
    query = transcribe_audio(audio_flat, model_size=model_size)
    
    # Step 4: Return the text
    return query


if __name__ == "__main__":
    # ─── Test: Run the STT module standalone ───
    print("=== Speech-to-Text Test ===")
    print("This will record 5 seconds of audio from your microphone.\n")
    
    query = speech_to_text(duration=5, model_size="base")
    
    print(f"\n--- Final Result ---")
    print(f"Transcribed query: {query}")
