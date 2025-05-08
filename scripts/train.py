import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AudioCNN  # Import your CNN model
from dataloader import train_loader, val_loader  # Import your DataLoaders

# Set device (this can be outside the if block)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training hyperparameters (these can also be outside)
num_epochs = 100  # Increase max epochs to allow for early stopping
patience = 10     # Number of epochs to wait for improvement

if __name__ == '__main__':
    # Instantiate the model (INSIDE the if block)
    num_emotions = 8
    model = AudioCNN(num_emotions).to(device)

    # Define loss function and optimizer (INSIDE the if block)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop (INSIDE the if block)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")

        # Validation loop (INSIDE the if block)
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item() * val_inputs.size(0)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total_samples += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

        val_epoch_loss = val_loss / val_total_samples
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

        # Early stopping check (INSIDE the if block)
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_audio_cnn_model.pth") # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    print("Finished Training (with early stopping)")