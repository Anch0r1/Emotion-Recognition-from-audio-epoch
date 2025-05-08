import torch
import torch.nn as nn
import librosa
import numpy as np
from prosodytrain import ProsodyModel

# Emotion mapping (must match training)
emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def extract_prosody_features(audio_path, sr=22050, n_fft=2048, hop_length=512, max_length=500):
    audio, sr = librosa.load(audio_path, sr=sr)

    f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=hop_length)
    f0 = librosa.util.normalize(np.nan_to_num(f0))

    rmse = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    rmse = librosa.util.normalize(np.nan_to_num(rmse))

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    tempo = (tempo - 60) / 60

    f0 = np.pad(f0, (0, max(0, max_length - len(f0))), 'constant')[:max_length]
    rmse = np.pad(rmse, (0, max(0, max_length - len(rmse))), 'constant')[:max_length]

    prosody_vector = np.concatenate([f0, rmse, np.array([tempo])])
    return torch.tensor(prosody_vector, dtype=torch.float32)

def predict_emotion(model, audio_path, device):
    model.eval()
    features = extract_prosody_features(audio_path).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(features)
        predicted_idx = torch.argmax(output, dim=1).item()
        return emotion_labels[predicted_idx]

if __name__ == "__main__":
    audio_path = r"C:\Users\Arnav\emrecogAudio\Specialtesting\shaanth1.wav"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Must match your trained model
    input_size = 500 * 2 + 1
    hidden_size = 128
    num_classes = 8

    model = ProsodyModel(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load("prosody_model.pth", map_location=device))

    predicted_emotion = predict_emotion(model, audio_path, device)
    print(f"Predicted Emotion: {predicted_emotion}")
