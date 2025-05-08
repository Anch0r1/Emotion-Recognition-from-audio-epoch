import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from fusion_dataset import FusionDataset
from feature_extractor_cnn import TruncatedAudioCNN
from prosody_feature_extractor import ProsodyFeatureExtractor
from fusion_model import FusionModel
from model import AudioCNN
from prosodytrain import ProsodyModel

# Hyperparameters
train_data_dir = "data/archive"
batch_size = 32
learning_rate = 0.001
num_epochs = 50
hidden_size_fusion = 256
num_emotions = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Fusion Model on device: {device}")

# Load trained CNN and wrap it as feature extractor
audio_cnn = AudioCNN(num_emotions=num_emotions)
audio_cnn.load_state_dict(torch.load("best_audio_cnn_model.pth", map_location=device))
audio_cnn.eval()
cnn_feature_extractor = TruncatedAudioCNN(audio_cnn).to(device).eval()

# Load trained Prosody model and wrap feature extractor
prosody_input_size = 1001  # f0 (500) + rmse (500) + tempo (1)
prosody_hidden_size = 128
prosody_model = ProsodyModel(prosody_input_size, prosody_hidden_size, num_emotions)
prosody_model.load_state_dict(torch.load("prosody_model.pth", map_location=device))
prosody_model.eval()
prosody_feature_extractor = ProsodyFeatureExtractor(prosody_model, feature_layer='fc1').to(device).eval()

# Build Fusion Dataset and DataLoaders
full_dataset = FusionDataset(train_data_dir, cnn_feature_extractor, prosody_feature_extractor, device)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Determine feature sizes
sample_combined_features, _ = next(iter(train_loader))
input_dim = sample_combined_features.shape[1]
cnn_feature_dim = 128  # Known output size of TruncatedAudioCNN
prosody_feature_dim = input_dim - cnn_feature_dim

# Define Fusion Model
fusion_model = FusionModel(cnn_feature_dim, prosody_feature_dim, hidden_size_fusion, num_emotions).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=learning_rate)

# Training Loop
best_val_accuracy = 0.0
for epoch in range(num_epochs):
    fusion_model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for combined_features, labels in train_loader:
        combined_features = combined_features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = fusion_model(combined_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * combined_features.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    # Validation
    fusion_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_features, val_labels in val_loader:
            val_features = val_features.to(device)
            val_labels = val_labels.to(device)
            val_outputs = fusion_model(val_features)
            val_loss += criterion(val_outputs, val_labels).item() * val_features.size(0)
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_accuracy = val_correct / val_total
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_accuracy:.4f}")

    if val_epoch_accuracy > best_val_accuracy:
        best_val_accuracy = val_epoch_accuracy
        torch.save(fusion_model.state_dict(), "best_fusion_model_fused.pth")
        print(f"Best validation accuracy updated: {best_val_accuracy:.4f}, saving model.")

# Final Save
torch.save(fusion_model.state_dict(), "fusion_model_fused_final.pth")
print("Final Fusion Model weights saved to fusion_model_fused_final.pth")
