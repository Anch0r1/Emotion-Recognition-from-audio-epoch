import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from load_prosody import create_prosody_dataloader
from prosodytrain import ProsodyModel  # Make sure this works as an import

def evaluate(model, val_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for prosody_batch, label_batch, _ in val_loader:
            prosody_batch = prosody_batch.to(device)
            label_batch = label_batch.to(device)
            outputs = model(prosody_batch)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    overall_accuracy = 100 * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")

    # Per-class precision, recall, and accuracy
    print("\nClassification Report (per-class metrics):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=2))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "data/archive"
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = create_prosody_dataloader(data_dir, batch_size=batch_size)

    input_size = val_loader.dataset[0][0].shape[0]
    hidden_size = 128
    num_classes = 8
    class_names = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

    model = ProsodyModel(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load("prosody_model.pth", map_location=device))

    evaluate(model, val_loader, device, class_names)
