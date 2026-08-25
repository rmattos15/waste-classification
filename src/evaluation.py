# This is the Module for evaluating the CNN Models including metrics, confusion matrices, inspection

import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Evaluates a trained model using the provided dataset.
# Returns the loss, accuracy, and the labels and predictions from the evaluation.
# The loss function is set to CrossEntropy by default.
# The device is set to Apple's MPS if available, otherwise CPU.
def evaluate_model(model, dataloader, loss_function=nn.CrossEntropyLoss(), device=device):
    correct = 0
    total = 0
    running_loss = 0.0

    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(images)

            predicted = torch.argmax(outputs, dim=1)

            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

            loss = loss_function(outputs, batch_labels)
            running_loss += loss.item()

            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    return loss, accuracy, all_labels, all_predictions

    
# Creates a confusion matrix to analyze the distribution of the actual classes and the prediction classes.
# The title is set to "Confusion Matrix" by default.
def create_confusion_matrix(labels, predictions, class_names, title="Confusion Matrix"):   
    plt.figure(figsize=(10, 9))
    
    cm = confusion_matrix(labels, predictions)
    
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    
    annotations = np.empty_like(cm, dtype=object)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
    
    sns.heatmap(cm, 
                annot=annotations,
                fmt="", 
                xticklabels=range(1,len(class_names)+1),
                yticklabels=class_names)
    
    plt.xlabel("Prediction", fontsize=13)
    plt.ylabel("Actual", fontsize=13)
    plt.title(title, fontsize=17, pad=20)
    plt.show()

# Prints the number of matches found and displays up to 10 examples.
# Mean and standard deviation are set to default values.
def inspecting_images(
    dataset,
    evaluation_data,
    labels,
    predictions,
    actual_class,
    predicted_class,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5)
):
    actual_class_index = dataset.class_to_idx[actual_class]
    predicted_class_index = dataset.class_to_idx[predicted_class]

    mistakes = [
        i for i, (actual, predicted)
        in enumerate(zip(labels, predictions))
        if actual == actual_class_index and predicted == predicted_class_index
    ]

    print(f"Found {len(mistakes)} {actual_class} → {predicted_class}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 5))

    for ax, i in zip(axes.flat, mistakes[:10]):
        image, label = evaluation_data[i]

        # Undo normalization for visualization
        mean_tensor = torch.tensor(mean).view(3, 1, 1)
        std_tensor = torch.tensor(std).view(3, 1, 1)

        image = image * std_tensor + mean_tensor
        image = image.permute(1, 2, 0)

        ax.imshow(image)
        ax.set_title(f"{actual_class} → {predicted_class}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()