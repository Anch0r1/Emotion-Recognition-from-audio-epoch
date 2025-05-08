import torch
import torch.nn as nn
from model import AudioCNN
from dataloader import test_loader
from sklearn.metrics import classification_report
import numpy as np

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model and load trained weights
    num_emotions = 8
    model = AudioCNN(num_emotions).to(device)
    model.load_state_dict(torch.load("best_audio_cnn_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Overall Accuracy
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    accuracy = correct / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Per-Class Accuracy Report
    emotion_labels = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]
    print("\nPer-Class Accuracy Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels, digits=4))
    import torch
import torch.nn as nn
from model import AudioCNN
from dataloader import test_loader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model and load trained weights
    num_emotions = 8
    model = AudioCNN(num_emotions).to(device)
    model.load_state_dict(torch.load("best_audio_cnn_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Overall Accuracy
    correct = np.sum(np.array(all_preds) == np.array(all_labels))
    accuracy = correct / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Per-Class Accuracy Report
    emotion_labels = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]
    print("\nPer-Class Accuracy Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=emotion_labels, yticklabels=emotion_labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()  # Ensure this line is present

if __name__ == '__main__':
    evaluate()
