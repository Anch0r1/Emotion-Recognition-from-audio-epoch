import torch
import torch.nn as nn
from prosodytrain import ProsodyModel  

class ProsodyFeatureExtractor(nn.Module):
    def __init__(self, original_prosody_model, feature_layer='fc1'):
        super(ProsodyFeatureExtractor, self).__init__()
        self.feature_layer_name = feature_layer
        self.original_model = original_prosody_model

    def forward(self, x):
        if self.feature_layer_name == 'fc1':
            return self.original_model.relu(self.original_model.fc1(x)) # Output after the first hidden layer's ReLU
        elif self.feature_layer_name == 'fc2':
            return self.original_model.relu(self.original_model.fc2(self.original_model.relu(self.original_model.fc1(x)))) # Output after the second hidden layer's ReLU (if it exists)
        else:
            raise ValueError(f"Feature layer '{self.feature_layer_name}' not found in the ProsodyModel.")

# if __name__ == '__main__':
#     # Example usage:
#     prosody_input_size = 179 # Adjust based on your prosody feature vector length
#     prosody_hidden_size = 128
#     num_emotions = 8
#     trained_prosody_model = ProsodyModel(prosody_input_size, prosody_hidden_size, num_emotions)
#     trained_prosody_model.load_state_dict(torch.load("prosody_model.pth", map_location='cpu')) # Load your trained prosody model weights
#     trained_prosody_model.eval()

#     feature_extractor = ProsodyFeatureExtractor(trained_prosody_model, feature_layer='fc1')
#     print("\nProsody Feature Extractor:")
#     for name, module in feature_extractor.named_modules():
#         print(f"  {name}: {module}")

#     # Example input (adjust shape as needed)
#     dummy_input_prosody = torch.randn(1, prosody_input_size)
#     prosody_features = feature_extractor(dummy_input_prosody)
#     print("Shape of Prosody features:", prosody_features.shape)
