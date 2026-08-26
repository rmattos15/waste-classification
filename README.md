# Waste Classification using Convolutional Neural Networks

## Project Summary 

Waste image classification using custom CNNs and Transfer Learning with ResNet-18 and EfficientNet-B0. Explores data augmentation, model capacity, learning rate scheduling and compares six CNN models/experiments, achieving 92.53% test accuracy with EfficientNet-B0.

## Overview/Motivation

Waste awareness, recycling and separation are probably more widespread than ever before. Although people in the past sorted things based on personal survival or monetary value, we see today a standardized and structured approach on how to dispose our trash. Some items may seem quite simple to identify, such as metal cans, glass bottles or food organics. Nonetheless, you may have already been in a situation where you couldn't properly decide which category best fits an item.

But what if we could have a more informed classification on this? This was the premise behind this project. To achieve this, we would apply Convolutional Neural Networks to a Waste Classification dataset and analyze different models. Particularly, we would not only see how they perform but rather try to understand what is actually happening in each case.

## Dataset

The dataset used in this project is the "Waste Classification", obtained from [Kaggle](https://www.kaggle.com/datasets/adithyachalla/waste-classification/data). It is described on Kaggle as containing 4000 images across nine household waste categories; however, our analysis found 4817 images in the downloaded dataset.

These categories are described on the dataset page as:

1. Cardboard : "cardboard boxes, packaging, and sheets."
2. Food Organics: "organic waste such as fruits and food."
3. Glass: "glass bottles, jars, and broken glass items."
4. Metal: "metal cans, foils, and scrap metal."
5. Miscellaneous Trash: "mixed or unclear waste types not fitting others."
6. Paper: "paper items like newspapers, sheets, books."
7. Plastic: "plastic containers, bags, and packaging."
8. Textile Trash: "clothes, fabric pieces, and textiles."
9. Vegetation: "leaves, branches, and other green waste."

Each image is stored directly under its respective folder and is named with arbitrary file names.
It is worth mentioning that the dataset is said to be "already organized and labeled". Still, we do a proper Exploratory Data Analysis before considering any CNN model. 

## Approach

We build a total of four custom CNN experiments, applying different techniques and changes in model capacity. Additionally, we created two transfer learning models using ResNet-18 and EfficientNet-B0.

A brief description of each model is shown below:

| Model        | Main Experiment                               |
|--------------|-----------------------------------------------|
| Baseline     | Initial custom CNN used as the baseline       |
| Augmented    | Data augmentation applied to the Baseline     |
| Upgraded     | Increased model capacity applied to Augmented |
| Scheduler    | Learning-rate scheduling applied to Upgraded  |
| ResNet       | Transfer learning with ResNet-18              |
| EfficientNet | Transfer learning with EfficientNet-B0        |

## Results

### Validation Results

| Model | Validation Loss | Validation Accuracy |
|-------|------------------:|----------------------:|
| Baseline | 2.8811 | 55.33% |
| Augmented | 1.2270 | 57.12% |
| Upgraded | 1.0630 | 62.10% |
| Scheduler | 1.3569 | 50.62% |
| ResNet | 0.3771 | 90.73% |
| EfficientNet | 0.2280 | 91.70% |

### Final Model

The best-performing model was EfficientNet-B0, which was therefore selected for evaluation on the test set.

| Model | Test Loss | Test Accuracy | Test Precision | Test Recall | Test F1 Score |
|-------|-----------:|---------------:|-----------------:|-------------:|----------------:|
| EfficientNet | 0.2682 | 92.53% | 0.9265 | 0.9253 | 0.9251 |

## Key Findings

Some of the key findings from this project include:

- Data augmentation provided a modest improvement in accuracy over the Baseline, along with a more significant improvement in validation loss.
- Increasing model capacity produced a larger improvement in validation performance.
- Learning-rate scheduling did not improve the Upgraded model in this project and instead reduced its validation performance.
- Transfer learning produced a substantial jump in performance compared with the custom CNNs.
- EfficientNet-B0 achieved the best validation and test performance among all models tested.
- Misclassification analysis revealed recurring confusion between visually similar categories, particularly Plastic with Metal and Glass.

> **Note:** More detailed analysis and discussion of these findings can be found throughout the Jupyter notebooks.

### Repository Structure

This repository is structured into different folders, each covering a different part of the project. Although each folder serves a specific purpose, they are all connected through the overall workflow of the project. A breakdown of each folder can be found below:

1. The `models` folder contains the trained weights for all models used during evaluation. Each model has its own `.pth` file, which can be loaded and evaluated directly.

2. The `notebooks` folder contains all Jupyter notebooks used for Exploratory Data Analysis, model preparation and results evaluation. The notebooks are numbered according to the suggested order of execution, but each notebook can also be run independently.

3. The `src` folder contains the Python modules used across the Jupyter notebooks.

4. The `training_errors` folder contains `.txt` files with the training loss of each model across the 50 training epochs. These files provide an additional way to understand how the training process progressed for each model.


## How to Run

To run the project, you will need all the required folders along with the dataset. The dataset is not included in this repository due to its overall file size, and also because it can be easily found and downloaded online [here](https://www.kaggle.com/datasets/adithyachalla/waste-classification/data).

After downloading the dataset, create a `data` folder containing the `cnn_waste_classification` folder. The expected structure is:

```text
data/
└── cnn_waste_classification/
    ├── 1-Cardboard/
    ├── 2-Food Organics/
    ├── ...
    └── 9-Vegetation/
```

This allows the dataset to be accessed using the path `data/cnn_waste_classification/`. If your dataset is stored in a different location, make sure to update the corresponding paths used when loading and saving data.
 
## Jupyter Working Directory

Due to a Jupyter configuration, the notebooks may open with the notebooks folder as the current working directory rather than the project root. This is why the following code appears at the beginning of most notebooks:

```python
%cd .. 
```

This command moves the current working directory back to the project root. Before running the notebooks, it is recommended to check the current working directory to avoid path-related issues.

## Requirements

The project was developed using Python and PyTorch. The main tools and libraries used include:

- PyTorch / Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- JupyterLab
