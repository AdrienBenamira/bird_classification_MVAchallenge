import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
import torch.nn as nn

nclasses = 20

class ResNet18_fineTune(nn.Module):
    def __init__(self):
        super(ResNet18_fineTune, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, nclasses)

    def forward(self, x):
        return self.model(x)


