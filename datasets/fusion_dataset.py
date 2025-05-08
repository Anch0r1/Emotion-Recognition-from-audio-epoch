import torch
from torch.utils.data import Dataset
from updated_emotion_dataset import ProsodyOnlyDataset
from emotion_dataset import EmotionAudioDataset
from prosodytrain import ProsodyModel
from model import AudioCNN
from feature_extractor_cnn import TruncatedAudioCNN
from prosody_feature_extractor import ProsodyFeatureExtractor

class FusionDataset(Dataset):
    def __init__(self, data_dir, cnn_feature_extractor, prosody_feature_extractor, device):
        self.spectrogram_dataset = EmotionAudioDataset(data_dir)
        self.prosody_dataset = ProsodyOnlyDataset(data_dir)
        self.cnn_feature_extractor = cnn_feature_extractor.to(device).eval()
        self.prosody_feature_extractor = prosody_feature_extractor.to(device).eval()
        self.device = device

        assert len(self.spectrogram_dataset) == len(self.prosody_dataset), \
            "Spectrogram and Prosody datasets must have the same number of samples."

    def __len__(self):
        return len(self.spectrogram_dataset)

    def __getitem__(self, idx):
        spectrogram, numerical_label = self.spectrogram_dataset[idx]
        spectrogram = spectrogram.unsqueeze(0).to(self.device)

        prosody_features, _, _ = self.prosody_dataset[idx]
        prosody_features = prosody_features.unsqueeze(0).to(self.device)

        with torch.no_grad():
            cnn_features = self.cnn_feature_extractor(spectrogram).squeeze(0).cpu()
            prosody_features_extracted = self.prosody_feature_extractor(prosody_features).squeeze(0).cpu()

        combined_features = torch.cat((cnn_features, prosody_features_extracted), dim=0)
        return combined_features, numerical_label

if __name__ == '__main__':
    train_data_dir = "data/archive"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained CNN and wrap as Truncated feature extractor
    audio_cnn = AudioCNN(num_emotions=8)
    audio_cnn.load_state_dict(torch.load("best_audio_cnn_model.pth", map_location=device))
    audio_cnn.eval()

    cnn_feature_extractor = TruncatedAudioCNN(audio_cnn).to(device).eval()

    # Load trained prosody model and wrap feature extractor
    prosody_input_size = 1001  # f0 (500) + rmse (500) + tempo (1)
    prosody_hidden_size = 128
    num_emotions = 8

    prosody_model = ProsodyModel(prosody_input_size, prosody_hidden_size, num_emotions)
    prosody_model.load_state_dict(torch.load("prosody_model.pth", map_location=device))
    prosody_model.eval()

    prosody_feature_extractor = ProsodyFeatureExtractor(prosody_model, feature_layer='fc1').to(device).eval()

    # Build FusionDataset and test
    fusion_dataset = FusionDataset(train_data_dir, cnn_feature_extractor, prosody_feature_extractor, device)
    fusion_loader = torch.utils.data.DataLoader(fusion_dataset, batch_size=4, shuffle=True)

    for combined_features, labels in fusion_loader:
        print("Combined features shape:", combined_features.shape)  # Should be [4, 256]
        print("Labels shape:", labels.shape)
        break
