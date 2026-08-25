# This is the Module for training the CNN models

import time
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Trains a CNN model
# The loss function is set to CrossEntropy by default.
# The device is set to Apple's MPS if available, otherwise CPU.
def training_model(
    model,
    train_loader,
    optimizer,
    loss_function=nn.CrossEntropyLoss(),
    device=device,
    epochs=50
):
    start_time = time.time()

    model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f"Training Epoch {epoch}:")
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        print(f"Loss: {epoch_loss:.4f}")

    total_time = time.time() - start_time

    print(f"Model Training took: {total_time / 60:.2f} minutes")