# wave2vecscript.py
# A script to perform continuous emotion recognition from microphone input using a pre-trained Wav2Vec2 model.  




import torch
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import time

# Path to your local model folder (where config.json, pytorch_model.bin, preprocessor_config.json exist)
MODEL_PATH = "/Users/mac/Downloads/hugging_face"

# Load feature extractor + model
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

# Emotion labels
id2label = {
    0: 'angry',
    1: 'calm',
    2: 'disgust',
    3: 'fearful',
    4: 'happy',
    5: 'neutral',
    6: 'sad',
    7: 'surprised'
}

def record_audio(duration=4, fs=16000):
    """Record audio for a fixed duration from microphone"""
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def predict_emotion(audio):
    """Run audio through model and return predicted emotion + confidence"""
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], probs[pred_id]

if __name__ == "__main__":
    print("🎤 Continuous Emotion Recognition (Press Ctrl+C to stop)")
    try:
        while True:
            audio = record_audio(duration=4)
            emotion, confidence = predict_emotion(audio)
            print(f"🔊 Detected Emotion: {emotion} ({confidence*100:.2f}% confidence)")
            time.sleep(0.5)  # small pause before next recording
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
