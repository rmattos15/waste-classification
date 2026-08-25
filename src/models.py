# This is the Module for the CNN models including all CNN architectures + pretrained models

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

# Baseline CNN used as the reference model for subsequent experiments.
# The network contains four convolutional layers, each followed by 2x2 max pooling. 
# The number of channels is kept relatively low to maintain a simple model and reduce computational cost.

# Conv1: (3,224,224) -> (6,222,222)
# Pool:  (6,222,222) -> (6,111,111)

# Conv2: (6,111,111) -> (12,109,109)
# Pool:  (12,109,109) -> (12,54,54)

# Conv3: (12,54,54) -> (24,52,52)
# Pool:  (24,52,52) -> (24,26,26)

# Conv4: (24,26,26) -> (48,24,24)
# Pool:  (48,24,24) -> (48,12,12)

class BaselineConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3) 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3) 
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3) 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)     

        # Fully Connected Layer (48*12*12 = 6912)
        self.fc1 = nn.Linear(in_features=48*12*12, out_features=90)
        self.fc2 = nn.Linear(in_features=90, out_features=9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Upgraded CNN with increased channel capacity compared to the Baseline Model.
# The network contains four convolutional layers, each followed by 2x2 max pooling. 
# The network has the same overall architecture as the baseline, 
# but uses more channels to increase model capacity.

# Conv1: (3,224,224) -> (16,222,222)
# Pool:  (16,222,222) -> (16,111,111)

# Conv2: (16,111,111) -> (32,109,109)
# Pool:  (32,109,109) -> (32,54,54)

# Conv3: (32,54,54) -> (64,52,52)
# Pool:  (64,52,52) -> (64,26,26)

# Conv4: (64,26,26) -> (128,24,24)
# Pool:  (128,24,24) -> (128,12,12)

class UpgradedConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) 
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3) 

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Fully Connected Layer (128*12*12 = 18432)
        self.fc1 = nn.Linear(in_features=128*12*12, out_features=90)
        self.fc2 = nn.Linear(in_features=90, out_features=9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Returns a pretrained ResNet-18 adapted for the waste classification task.
# The number of classes is set to 9 by default.
def get_resnet18(num_classes=9):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


# Returns a pretrained EfficientNet-B0 adapted for the waste classification task.
# The number of classes is set to 9 by default.
def get_efficientnet_b0(num_classes=9):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    
    return model
