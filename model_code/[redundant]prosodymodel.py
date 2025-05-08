import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import create_dataloaders  # Assuming this is the correct import

# Define the neural network model
class ProsodyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ProsodyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer with num_classes output units

    def forward(self, x):
        out = self.fc1(x)  # Pass input through the first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Pass through the second layer (final output)
        return out


# Define the training loop
def train_prosody(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0  # Track the loss for this epoch

        for prosody_batch, numerical_label_batch, _ in train_loader:
            prosody_batch, numerical_label_batch = prosody_batch.to(device), numerical_label_batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(prosody_batch)
            loss = criterion(outputs, numerical_label_batch)  # Calculate the loss

            # Backward pass and optimization
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)  # Calculate average loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation (optional)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0

            for prosody_batch, numerical_label_batch, _ in val_loader:
                prosody_batch, numerical_label_batch = prosody_batch.to(device), numerical_label_batch.to(device)
                outputs = model(prosody_batch)
                loss = criterion(outputs, numerical_label_batch)

                val_loss += loss.item()

                # Convert outputs to predicted labels
                _, predicted = torch.max(outputs, 1)
                total += numerical_label_batch.size(0)
                correct += (predicted == numerical_label_batch).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save the model periodically
        torch.save(model.state_dict(), "prosody_model.pth")


# Main function to initialize everything and start training
# Main function to initialize everything and start training
if __name__ == "__main__":
    # Set the data directory and batch size
    train_data_dir = "data/archive"  # Your dataset path
    batch_size = 4
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloaders for train and validation sets
    train_loader, val_loader = create_dataloaders(train_data_dir, batch_size=batch_size)

    # The prosody feature vector length is 1000 (since we padded it in the dataset)
    prosody_input_size = 1000  # This should match the fixed length of your input features
    prosody_hidden_size = 128  # Number of neurons in the hidden layer
    num_emotions = 8  # Number of emotion classes
    prosody_model = ProsodyModel(prosody_input_size, prosody_hidden_size, num_emotions).to(device)

    # Define loss function and optimizer
    prosody_criterion = nn.CrossEntropyLoss()  # Loss function for classification
    prosody_optimizer = optim.Adam(prosody_model.parameters(), lr=0.001)  # Adam optimizer

    # Train the prosody model
    train_prosody(prosody_model, train_loader, val_loader, prosody_criterion, prosody_optimizer, num_epochs, device)
