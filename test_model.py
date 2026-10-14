
from transformers import pipeline
print("Loading speech emotion recognition model...")
model_id = "Dpngtm/wav2vec2-emotion-recognition"

classifier = pipeline(
    "audio-classification",
    model=model_id
)

print("Model loaded successfully!")
print("Emotion labels:", classifier.model.config.id2label)
