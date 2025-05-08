import torch
from updated_emotion_dataset import ProsodyOnlyDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Custom collate function for padding sequences
def prosody_collate_fn(batch):
    prosody_tensors, numerical_labels, string_labels = zip(*batch)
    # Padding the prosody sequences (f0, rmse, tempo) to the same length
    padded_prosody = pad_sequence(prosody_tensors, batch_first=True)
    numerical_labels = torch.stack(numerical_labels)  # Stack numerical labels into a tensor
    return padded_prosody, numerical_labels, list(string_labels)

# Function to create DataLoader
def create_prosody_dataloader(data_dir, batch_size=4, validation_split=0.2):
    # Create dataset
    dataset = ProsodyOnlyDataset(data_dir=data_dir)

    # Calculate sizes for train and validation sets
    train_size = int((1 - validation_split) * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoader for training and validation sets with the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=prosody_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=prosody_collate_fn)

    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    data_dir = "data/archive"  # Path to your dataset
    batch_size = 4

    # Create the dataloaders
    train_loader, val_loader = create_prosody_dataloader(data_dir, batch_size=batch_size)

    # To check the data loading and output the shapes of the batches
    for prosody_batch, numerical_label_batch, string_label_batch in train_loader:
        print(f"Prosody batch shape: {prosody_batch.shape}")
        print(f"Numerical label batch: {numerical_label_batch}")
        print(f"String label batch: {string_label_batch}")
        break  # Only print one batch for demonstration
