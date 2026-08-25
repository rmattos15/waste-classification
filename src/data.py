# This is the Module for data operations including dataset loading, splits, transforms, loaders

import torchvision.transforms as transforms

import numpy as np
from pathlib import Path

from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

DATA_DIR = Path("data/cnn_waste_classification")

# Loads the dataset using ImageFolder and applies the specified transform.
def get_data(transform):
    return ImageFolder(DATA_DIR, transform=transform)

# Returns the basic preprocessing transform used for the custom CNN models.
# No data augmentation is applied here.
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))
    ])

# Returns the augmented preprocessing transform used for the custom CNN models.
def get_transform_aug():
    return transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(200),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

# Returns the augmented preprocessing transform used for training the transfer CNN models.
# The images are normalized following the ImageNet dataset normalization
def get_transform_transfer_train():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

# Returns the augmented preprocessing transform used for evaluating the transfer CNN models.
# The images are normalized following the ImageNet dataset normalization
def get_transform_transfer_eval():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ]) 

# Returns the indices for the train/validation/test sets,
# defined through random splits of the dataset.
# Split is 70/15/15.
# The random state seed is set to 15 by default.
def get_data_splits(dataset, random_state=15):
    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=random_state
    )

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=labels[temp_indices],
        random_state=random_state
    )
    
    return train_indices, val_indices, test_indices

# Generates the datasets for train/validation/test.
# Train uses augmentation while validation/test don't.
# The random state seed is set to 15 by default.
def get_datasets(train_transform, val_transform, data_dir=DATA_DIR, random_state=15):
    base_dataset = ImageFolder(data_dir)
    
    train_indices, val_indices, test_indices = get_data_splits(
        base_dataset, random_state
    )

    train_dataset = ImageFolder(data_dir, transform=train_transform)
    val_dataset = ImageFolder(data_dir, transform=val_transform)
    test_dataset = ImageFolder(data_dir, transform=val_transform)
    
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)
    test_data = Subset(test_dataset, test_indices)

    return train_data, val_data, test_data

# Prepares the DataLoader for each one of the sets
# The batch size is set to 32 by default.
def get_dataloaders(train_data, val_data, test_data, batch_size=32):
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

