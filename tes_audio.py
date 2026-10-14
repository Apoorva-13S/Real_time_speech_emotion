
import librosa
from transformers import pipeline

model_id = "Dpngtm/wav2vec2-emotion-recognition"

print("Loading model...")
classifier = pipeline("audio-classification", model=model_id)

audio_file = input("Enter the full path to your audio file: ")

print("Analyzing audio...")
audio, sr = librosa.load(audio_file, sr=16000, mono=True)

results = classifier(
    {"raw": audio, "sampling_rate": sr},
    top_k=7
)

print("\nEmotion predictions:")
for result in results:
    print(f"{result['label']}: {result['score'] * 100:.2f}%")
