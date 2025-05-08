from torch.utils.data import DataLoader
from emotion_dataset import EmotionAudioDataset

# Change path to point to the archive folder
dataset = EmotionAudioDataset("data/archive")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for spectrograms, labels in dataloader:
    print("Batch shape:", spectrograms.shape)  # [B, 1, 128, 216]
    print("Labels:", labels)
    break
