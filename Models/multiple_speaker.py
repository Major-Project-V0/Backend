import torch
import sounddevice as sd
import numpy as np
import librosa
import scipy.signal as signal

print("Loading Silero VAD model...")
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)

(get_speech_timestamps, _, read_audio, _, collect_chunks) = utils

TARGET_SR = 16000
CHUNK = 0.5   # seconds
OVERLAP_THRESHOLD = 1.35  # harmonic complexity threshold

mic_sr = int(sd.query_devices(sd.default.device['input'], 'input')['default_samplerate'])
block_size = int(mic_sr * CHUNK)

print(f"Mic SR: {mic_sr}")
print("System Ready...\n")


def detect_speech(audio16):
    ts = get_speech_timestamps(torch.tensor(audio16), model, sampling_rate=TARGET_SR)
    return len(ts) > 0


def harmonic_complexity(audio16):
    """Detect overlapping voices using harmonic peak density."""
    f, t, Zxx = signal.stft(audio16, TARGET_SR, nperseg=512)
    mag = np.abs(Zxx)

    # Sum spectrum across time (gives frequency energy)
    spectrum = np.mean(mag, axis=1)

    # Detect harmonic peaks
    peaks, _ = signal.find_peaks(spectrum, height=np.max(spectrum)*0.2, distance=20)

    return len(peaks)


def callback(indata, frames, time_info, status):
    audio = indata[:, 0].astype(np.float32)

    # resample to 16k
    audio16 = librosa.resample(audio, orig_sr=mic_sr, target_sr=TARGET_SR)

    # Energy check
    if np.mean(audio16**2) < 1e-5:
        print("🔇 No speech")
        return

    # Speech or not?
    if not detect_speech(audio16):
        print("🔇 No speech")
        return

    # Measure harmonic peaks
    h = harmonic_complexity(audio16)

    if h > OVERLAP_THRESHOLD*2 :      # dynamic threshold
        print("🚨 MULTIPLE VOICES DETECTED!")
    else:
        print("🟢 Single speaker")


with sd.InputStream(callback=callback, channels=1, samplerate=mic_sr, blocksize=block_size):
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("Program terminated.")
