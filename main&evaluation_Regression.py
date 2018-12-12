import os
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

dictionnaire_correspondance={4:0, 9:1,10:2,11:3,12:4,13:5,14:6,15:7,16:8,19:9,20:10,21:11,23:12,26:13,28:14,29:15,
                             30:16,31:17,33:18,34:19}

print("Start load data")
features_train = np.load(os.path.join(config["load_dir"]+ '/_feature_train_assigment.npy'))
labels_train = np.load(os.path.join(config["load_dir"]+ '/_label_train_assigment.npy'))
features_val = np.load(os.path.join(config["load_dir"]+ '/_feature_val_assigment.npy'))
features_test = np.load(os.path.join(config["load_dir"]+ '/_feature_test_assigment.npy'))
labels_val = np.load(os.path.join(config["load_dir"]+ '/_label_val_assigment.npy'))
features_train_crop = np.load(os.path.join(config["load_dir"]+ '/_feature_train_assigment_crop.npy'))
features_val_crop = np.load(os.path.join(config["load_dir"]+ '/_feature_val_assigment_crop.npy'))
features_test_crop = np.load(os.path.join(config["load_dir"]+ '/_feature_test_assigment_crop.npy'))


if config["concatenate"]:
    features_train = np.concatenate((features_train, features_train_crop), axis=1)
    features_val = np.concatenate((features_val, features_val_crop), axis=1)
    features_test = np.concatenate((features_test, features_test_crop), axis=1)


print("End load data")
print("Start regression")
LR = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=100)
LR.fit(features_train, labels_train)
print("End regression")

labels_pred = LR.predict(features_val)
num_class = 20
acc = np.zeros((num_class, num_class), dtype=np.float32)
for i in range(len(labels_val)):
    acc[dictionnaire_correspondance[labels_val[i]], dictionnaire_correspondance[labels_pred[i]]] += 1.0

print('Accuracy on Validation: %f' % (sum([acc[i,i] for i in range(num_class)]) / len(labels_val)))

print("Start Predict and create file csv")
labels_pred = LR.predict(features_test)
df = pd.read_csv('kaggle_template.csv')
for dossier, sous_dossiers, fichiers in os.walk('bird_dataset/test_images/mistery_category'):
    for num, fichier in enumerate(fichiers):
        num_photo = df.loc[df['Id'] == fichier.split('.')[0]].index[0]
        df.Category[num_photo] = dictionnaire_correspondance[int(labels_pred[num])]
print("Succesfully wrote kaggle.csv, you can upload this file to the kaggle competition website")

df.to_csv('kaggle.csv',index=False)

