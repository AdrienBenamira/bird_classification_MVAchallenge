from torchvision import datasets
from models.model_finetunning import ResNet18_fineTune
import os
import torch
import torch.optim as optim
from train import train_model
import torch.nn as nn
import torch.utils.data as utils
import json
import matplotlib.pyplot as plt

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

use_cuda = torch.cuda.is_available()
torch.manual_seed(config["seed"])

if not os.path.isdir(config["experiment"]):
    os.makedirs(config["experiment"])


# Data initialization and loading
from data import data_transforms

data_dir = config["data_dir"]
# loading datasets with PyTorch ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train_images', 'val_images']}
# defining data loaders to load data using image_datasets and transforms, here we also specify batch size for the mini batch
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config["batchsize"],
                                             shuffle=True, num_workers=4)
              for x in ['train_images', 'val_images']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train_images', 'val_images']}
class_names = image_datasets['train_images'].classes

model = ResNet18_fineTune()

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
criterion = nn.CrossEntropyLoss()

# Run the functions and save the best model in the function model_ft.
model_ft, losses_train, accuracy_train, losses_val, accuracy_val = train_model(model, criterion, optimizer,
                                                                               dataloders, dataset_sizes, use_cuda,
                                                                               num_epochs=config["epochs"])

plt.figure(1)
plt.subplot(221)
plt.plot(losses_train)
plt.ylabel('Losses Train')
plt.subplot(222)
plt.plot(accuracy_train)
plt.ylabel('Accuracy Train')
plt.subplot(223)
plt.plot(losses_val)
plt.ylabel('Losses Eval')
plt.subplot(224)
plt.plot(accuracy_val)
plt.ylabel('Accuracy Eval')
plt.show()
