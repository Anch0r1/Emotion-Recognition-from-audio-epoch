import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, cnn_feature_dim, prosody_feature_dim, hidden_size, num_classes):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(cnn_feature_dim + prosody_feature_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    # Example usage:
    cnn_feature_dim = 512 # Adjust based on your CNN's output
    prosody_feature_dim = 128 # Adjust based on your Prosody Feature Extractor's output
    hidden_size = 256
    num_emotions = 8
    fusion_model = FusionModel(cnn_feature_dim, prosody_feature_dim, hidden_size, num_emotions)
    print("Fusion Model:")
    print(fusion_model)

    dummy_combined_features = torch.randn(1, cnn_feature_dim + prosody_feature_dim)
    output = fusion_model(dummy_combined_features)
    print("Shape of Fusion Model output:", output.shape)