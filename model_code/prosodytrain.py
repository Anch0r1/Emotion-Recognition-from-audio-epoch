import torch
import torch.nn as nn
import torch.optim as optim
from load_prosody import create_prosody_dataloader

# Updated Prosody Model with Dropout & BatchNorm
class ProsodyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ProsodyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def train_prosody(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for prosody_batch, numerical_label_batch, _ in train_loader:
            prosody_batch, numerical_label_batch = prosody_batch.to(device), numerical_label_batch.to(device)
            optimizer.zero_grad()
            outputs = model(prosody_batch)
            loss = criterion(outputs, numerical_label_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_loss:.4f}", end='  ')

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0

            for prosody_batch, numerical_label_batch, _ in val_loader:
                prosody_batch, numerical_label_batch = prosody_batch.to(device), numerical_label_batch.to(device)
                outputs = model(prosody_batch)
                loss = criterion(outputs, numerical_label_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == numerical_label_batch).sum().item()
                total += numerical_label_batch.size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"Val Loss: {avg_val_loss:.4f}  Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "prosody_model.pth")
    print("✅ Model saved as prosody_model.pth")


# Main entry
if __name__ == "__main__":
    train_data_dir = "data/archive"
    batch_size = 4
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Training on device: {device}")

    train_loader, val_loader = create_prosody_dataloader(train_data_dir, batch_size=batch_size)

    prosody_input_size = train_loader.dataset[0][0].shape[0]
    prosody_hidden_size = 128
    num_emotions = 8

    model = ProsodyModel(prosody_input_size, prosody_hidden_size, num_emotions).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_prosody(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
