
import sounddevice as sd

print("Available audio devices:")
print(sd.query_devices())

print("\nTesting microphone for 5 seconds...")
print("Speak normally!")

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="float32"
)

sd.wait()

print("\nRecording finished!")
print("Audio shape:", audio.shape)
print("Maximum audio level:", abs(audio).max())
