# updated_emotion_dataset.py

from torch.utils.data import Dataset
import os
import librosa
import numpy as np
import torch

emotion_map = {
    '01': (0, 'neutral'), '02': (1, 'calm'), '03': (2, 'happy'), '04': (3, 'sad'),
    '05': (4, 'angry'), '06': (5, 'fearful'), '07': (6, 'disgust'), '08': (7, 'surprised')
}

class ProsodyOnlyDataset(Dataset):
    def __init__(self, data_dir, sr=22050, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.file_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files if f.endswith(".wav")
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label_id = int(os.path.basename(audio_path).split('-')[2])
        label_str = f"{label_id:02d}"
        label, emotion_label = emotion_map[label_str]

        audio, sr = librosa.load(audio_path, sr=self.sr)

        f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=self.hop_length)
        f0 = librosa.util.normalize(np.nan_to_num(f0))

        rmse = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        rmse = librosa.util.normalize(np.nan_to_num(rmse))

        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        tempo = (tempo - 60) / 60

        max_length = 500  # Shortened to reduce input size
        f0 = np.pad(f0, (0, max(0, max_length - len(f0))), 'constant')[:max_length]
        rmse = np.pad(rmse, (0, max(0, max_length - len(rmse))), 'constant')[:max_length]

        prosody_vector = np.concatenate([f0, rmse, np.array([tempo])])
        prosody_tensor = torch.tensor(prosody_vector, dtype=torch.float32)

        return prosody_tensor, torch.tensor(label, dtype=torch.long), emotion_label
