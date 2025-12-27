import numpy as np
import sounddevice as sd
import librosa
import noisereduce as nr
import time
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import scipy.signal as signal


# ============================================================
#  LOAD WAV2VEC2 MODEL + PROCESSOR
# ============================================================

MODEL_PATH = "wave2vec2_model"   # Your trained model folder

try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
    print("📦 Model loaded successfully!")
except Exception as e:
    print("\n❌ ERROR: Could not load model")
    print(e)
    exit()


EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprised"]


# ============================================================
#  AUDIO PREPROCESSOR — CRUCIAL FOR ACCURATE RESULTS
# ============================================================

def preprocess_audio(audio, sr=16000):
    """
    Matches microphone audio to dataset characteristics.
    Removes silence, normalizes volume, reduces noise,
    and performs slight pitch normalization.
    """

    # Convert to float32
    audio = audio.astype(np.float32)

    # -------------------------
    # 1️⃣ Remove silence
    # -------------------------
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # If silence only
    if len(audio) < 1000:
        return None

    # -------------------------
    # 2️⃣ Loudness Normalization
    # -------------------------
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # -------------------------
    # 3️⃣ Noise Reduction
    # -------------------------
    audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.9)

    # -------------------------
    # 4️⃣ Pitch Normalization
    #    Shifts pitch slightly so deep voices don’t break the model
    # -------------------------
    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1)

    # -------------------------
    # 5️⃣ Bandpass Filter (telephone-like, same as many datasets)
    # -------------------------
    b, a = signal.butter(4, [300/(sr/2), 3400/(sr/2)], btype='band')
    audio = signal.filtfilt(b, a, audio)

    # -------------------------
    # 6️⃣ Final normalization
    # -------------------------
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    return audio


# ============================================================
#  SPEECH / SILENCE DETECTION
# ============================================================

def compute_amplitude(audio):
    return float(np.max(np.abs(audio)))


def is_silent(audio, threshold=0.02):
    return compute_amplitude(audio) < threshold


# ============================================================
#  MICROPHONE TESTER
# ============================================================

def test_microphone(device_id, sr=16000):
    print("\n🎤 Testing microphone, speak NOW for 2 seconds...")

    try:
        audio = sd.rec(int(2 * sr), samplerate=sr, channels=1,
                       dtype="float32", device=device_id)
        sd.wait()
    except Exception as e:
        print("❌ Microphone error:", e)
        return False

    audio = audio.flatten()
    amp = compute_amplitude(audio)
    print(f"⭐ Test amplitude = {amp}")

    if amp < 0.02:
        print("❌ Mic not capturing enough sound.")
        return False

    print("✅ Microphone OK!")
    return True


# ============================================================
#  EMOTION PREDICTION
# ============================================================

def predict_emotion(clean_audio):
    inputs = processor(clean_audio, sampling_rate=16000,
                       return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    probs = torch.softmax(logits, dim=-1).numpy()[0]
    return probs


# ============================================================
#  MAIN PROGRAM
# ============================================================

def main():
    print("\n======================================")
    print("🔊 AVAILABLE AUDIO INPUT DEVICES")
    print("======================================")

    devices = sd.query_devices()

    input_ids = []
    for idx, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"{idx}: {d['name']}  (channels={d['max_input_channels']})")
            input_ids.append(idx)

    device_id = int(input("\n🎤 Enter microphone Device ID: "))
    print(f"\nUsing microphone ID {device_id}")

    # Mic test
    if not test_microphone(device_id):
        print("\n❌ Try another device.")
        exit()

    print("\n🎙 Speak now… (listening for 60 seconds)")
    print("----------------------------------------------------")

    SR = 16000
    FRAME = 5
    DURATION = 60

    emotion_scores = []
    start = time.time()

    while time.time() - start < DURATION:

        audio = sd.rec(int(FRAME * SR), samplerate=SR,
                       channels=1, dtype="float32", device=device_id)
        sd.wait()

        audio = audio.flatten()

        amp = compute_amplitude(audio)
        print(f"[DEBUG] Frame amplitude = {amp}")

        if is_silent(audio):
            print("[SKIPPED] Silence detected...")
            continue

        # Preprocess audio
        clean = preprocess_audio(audio, SR)
        if clean is None:
            print("[SKIPPED] Cleaned audio still too silent.")
            continue

        # Predict
        probs = predict_emotion(clean)
        emotion_scores.append(probs)

        print("\nFrame Emotion Probabilities:")
        for emo, p in zip(EMOTIONS, probs):
            print(f"  {emo:10s}: {p*100:.2f}%")

    if len(emotion_scores) == 0:
        print("\n❌ No speech detected. Try again.")
        return

    avg = np.mean(emotion_scores, axis=0)

    print("\n====================================")
    print("📊 FINAL AVERAGE EMOTION RESULTS")
    print("====================================")

    for emo, p in zip(EMOTIONS, avg):
        print(f"  {emo:10s}: {p*100:.2f}%")

    final_emotion = EMOTIONS[np.argmax(avg)]
    print("\n🧠 Dominant Emotion:", final_emotion.upper())


# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    main()
