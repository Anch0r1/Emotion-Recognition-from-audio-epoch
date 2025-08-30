import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from emotion_dataset import EmotionAudioDataset 


data_dir = "data/archive"  

# Instantiate the full dataset
full_dataset = EmotionAudioDataset(data_dir)

# Split the dataset into training, validation, and test sets with stratification
train_indices, temp_indices = train_test_split(
    range(len(full_dataset)), test_size=0.2, stratify=full_dataset.labels, random_state=42
)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, stratify=[full_dataset.labels[i] for i in temp_indices], random_state=42
)

# Create Subset datasets for each split
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

# Define batch size and number of workers
batch_size = 32
num_workers = 4  # Adjust based on your CPU cores

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

