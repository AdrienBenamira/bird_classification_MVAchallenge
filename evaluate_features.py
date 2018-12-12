import os
import numpy as np
import torch
from models.model import Net
import pandas as pd
import json


with open('config.json') as json_data_file:
    config = json.load(json_data_file)

use_cuda = torch.cuda.is_available()

state_dict = torch.load(config["model"])

df = pd.read_csv('kaggle_template.csv')


features_test = np.load(os.path.join(config["load_dir"] + '/_feature_test_assigment.npy'))
features_test_crop = np.load(os.path.join(config["load_dir"] + '/_feature_test_assigment_crop.npy'))
if config["concatenate"]:
    features_test = np.concatenate((features_test, features_test_crop), axis=1)

#Model
model = Net(features_test.shape[1])

model.load_state_dict(state_dict['model'])
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')


for dossier, sous_dossiers, fichiers in os.walk(config["test_dir"]):
    for num, fichier in enumerate(fichiers):
        num_photo = df.loc[df['Id'] == fichier.split('.')[0]].index[0]
        data = torch.tensor(features_test[num])
        if use_cuda:
            data = data.cuda()
        output = model(data)
        prout, pred = torch.max(output.data, 0)
        df.Category[num_photo] = pred

df.to_csv('kaggle.csv',index=False)
print("Succesfully wrote kaggle.csv, you can upload this file to the kaggle competition website")


