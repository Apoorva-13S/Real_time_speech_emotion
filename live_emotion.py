
import sounddevice as sd
import numpy as np
import librosa
from transformers import pipeline

# 1. Load the pretrained emotion recognition model
print("Loading emotion recognition model...")

classifier = pipeline(
    "audio-classification",
    model="Dpngtm/wav2vec2-emotion-recognition"
)

print("Model loaded successfully!")

# 2. Recording settings
SAMPLE_RATE = 16000
DURATION = 3  # Record 3 seconds at a time

print("\nLive Emotion Detection Started!")
print("Speak naturally into your microphone.")
print("Press Ctrl+C in the terminal to stop.\n")

try:
    while True:
        # 3. Record audio from the microphone
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        # 4. Convert stereo-shaped array to a 1D array
        audio = audio.flatten()

        # 5. Check whether the recording contains sound
        if np.max(np.abs(audio)) < 0.01:
            print("Very quiet audio. Please speak a little louder.")
            continue

        # 6. Predict emotions from the recorded audio
        results = classifier(
            {"raw": audio, "sampling_rate": SAMPLE_RATE}
        )

        # 7. Display the strongest prediction
        best = results[0]
        emotion = best["label"]
        score = best["score"] * 100

        print(f"Detected emotion: {emotion} ({score:.2f}%)")

except KeyboardInterrupt:
    print("\nLive emotion detection stopped.")
