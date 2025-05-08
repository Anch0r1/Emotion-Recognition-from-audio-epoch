import torch
import torch.nn as nn
from model import AudioCNN  # Your original trained model

class TruncatedAudioCNN(nn.Module):
    def __init__(self, original_model):
        super().__init__()

        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            nn.ReLU(),
            nn.MaxPool2d(2),

            original_model.conv2,
            original_model.bn2,
            nn.ReLU(),
            nn.MaxPool2d(2),

            original_model.conv3,
            original_model.bn3,
            nn.ReLU(),
            nn.MaxPool2d(2),

            original_model.conv4,
            original_model.bn4,
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            original_model.fc1,
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained AudioCNN
    full_model = AudioCNN(num_emotions=8).to(device)
    full_model.load_state_dict(torch.load("best_audio_cnn_model.pth", map_location=device))
    full_model.eval()

    # Build truncated version
    feature_extractor = TruncatedAudioCNN(full_model).to(device)
    feature_extractor.eval()

    # Test with dummy spectrogram input
    dummy_input = torch.randn(1, 1, 128, 216).to(device)
    with torch.no_grad():
        output = feature_extractor(dummy_input)
    print("CNN Feature Vector Shape:", output.shape)  # Expected: [1, 128]
