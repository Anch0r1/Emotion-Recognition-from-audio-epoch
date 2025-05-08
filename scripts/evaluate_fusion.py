import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fusion_dataset import FusionDataset
from fusion_model import FusionModel
from feature_extractor_cnn import TruncatedAudioCNN
from prosody_feature_extractor import ProsodyFeatureExtractor
from model import AudioCNN
from prosodytrain import ProsodyModel
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Settings ---
data_dir = "data/archive"
batch_size = 32
num_emotions = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Emotion labels ---
emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# --- Load feature extractors ---
audio_cnn = AudioCNN(num_emotions=num_emotions)
audio_cnn.load_state_dict(torch.load("best_audio_cnn_model.pth", map_location=device))
cnn_feature_extractor = TruncatedAudioCNN(audio_cnn).to(device).eval()

prosody_model = ProsodyModel(1001, 128, num_emotions)
prosody_model.load_state_dict(torch.load("prosody_model.pth", map_location=device))
prosody_feature_extractor = ProsodyFeatureExtractor(prosody_model, feature_layer="fc1").to(device).eval()

# --- Dataset and Loader ---
fusion_dataset = FusionDataset(data_dir, cnn_feature_extractor, prosody_feature_extractor, device)
val_size = int(0.2 * len(fusion_dataset))
_, val_dataset = torch.utils.data.random_split(fusion_dataset, [len(fusion_dataset) - val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Load trained fusion model ---
input_dim = next(iter(val_loader))[0].shape[1]
cnn_feat_dim = 128
prosody_feat_dim = input_dim - cnn_feat_dim
fusion_model = FusionModel(cnn_feat_dim, prosody_feat_dim, hidden_size=256, num_classes=num_emotions)
fusion_model.load_state_dict(torch.load("best_fusion_model_fused.pth", map_location=device))
fusion_model = fusion_model.to(device)
fusion_model.eval()

# --- Evaluation ---
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in val_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = fusion_model(features)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- Metrics ---
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nFusion Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=emotion_labels, digits=4))

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Fusion Model")
plt.tight_layout()
plt.show()
