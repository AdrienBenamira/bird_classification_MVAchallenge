import os
import torch
import torch.optim as optim
from models.model import Net
from train import train_model
import torch.nn as nn
import numpy as np
import torch.utils.data as utils
import json
import matplotlib.pyplot as plt
with open('config.json') as json_data_file:
    config = json.load(json_data_file)


use_cuda = torch.cuda.is_available()
torch.manual_seed(config["seed"])
if not os.path.isdir(config["experiment"]):
    os.makedirs(config["experiment"])


dictionnaire_correspondance={4:0, 9:1,10:2,11:3,12:4,13:5,14:6,15:7,16:8,19:9,20:10,21:11,23:12,26:13,28:14,29:15,
                             30:16,31:17,33:18,34:19}



features_train = np.load(os.path.join(config["load_dir"]+ '/_feature_train_assigment.npy'))
labels_train = np.load(os.path.join(config["load_dir"] + '/_label_train_assigment.npy'))
features_val = np.load(os.path.join(config["load_dir"] + '/_feature_val_assigment.npy'))
labels_val = np.load(os.path.join(config["load_dir"] + '/_label_val_assigment.npy'))
features_train_crop = np.load(os.path.join(config["load_dir"] + '/_feature_train_assigment_crop.npy'))
features_val_crop = np.load(os.path.join(config["load_dir"] + '/_feature_val_assigment_crop.npy'))

if config["concatenate"]:
    features_train = np.concatenate((features_train, features_train_crop), axis=1)
    features_val = np.concatenate((features_val, features_val_crop), axis=1)


my_x = features_train # a list of numpy arrays
my_y = labels_train # another list of numpy arrays (targets)
tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor([dictionnaire_correspondance[i]]) for i in my_y])
my_dataset_train = utils.TensorDataset(tensor_x,tensor_y) # create your datset
my_x = features_val # a list of numpy arrays
my_y = labels_val # another list of numpy arrays (targets)
tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
tensor_y = torch.stack([torch.Tensor([dictionnaire_correspondance[i]]) for i in my_y])
my_dataset_val = utils.TensorDataset(tensor_x,tensor_y) # create your datset

features_datasets = {'train_images': my_dataset_train, 'val_images':my_dataset_val}
dataloders = {x: torch.utils.data.DataLoader(features_datasets[x], batch_size=config["batchsize"],
                                             shuffle=True, num_workers=4)
              for x in ['train_images', 'val_images']}
dataset_sizes = {x: len(features_datasets[x]) for x in ['train_images', 'val_images']}


# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

model = Net(features_train.shape[1])

print(model)

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
