import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from typing import List  # Import List for type hinting

# Mapping RAVDESS emotion codes to string labels and integer indices
emotion_map = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
}
emotion_to_index = {name: idx for idx, name in enumerate(emotion_map.values())}


class EmotionAudioDataset(Dataset):
    def __init__(self, directory: str, sr: int = 22050, n_mels: int = 128, max_len: int = 216):
        """
        directory: folder containing .wav files
        sr: sampling rate
        n_mels: number of mel bands
        max_len: pad/truncate to fixed number of frames (time axis)
        """
        self.file_paths: List[str] = []  # Explicitly type file_paths as a list of strings
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".wav"):
                    self.file_paths.append(os.path.join(root, f))

        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len  # ensures uniform input size for CNNs
        self.labels: List[int] = [self.parse_emotion_from_filename(os.path.basename(fp)) for fp in self.file_paths]  # Calculate labels here

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        y, sr = librosa.load(file_path, sr=self.sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad or truncate to fixed time length
        if mel_db.shape[1] < self.max_len:
            pad_width = self.max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :self.max_len]

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # Convert to tensor: shape (1, mel, time)
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

        # Get label from pre-computed list
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return mel_tensor, label_tensor

    def parse_emotion_from_filename(self, filename: str) -> int:
        """Parse emotion code and return integer label index."""
        emotion_code = int(filename.split("-")[2])
        emotion = emotion_map[emotion_code]
        return emotion_to_index[emotion]