import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASS_MAPPING = {'COVID':0, 'Normal':1, 'Viral Pneumonia':2, 'Lung_Opacity':3}

def readData(path="./COVID-19_Radiography_Dataset/", classes=['COVID', 'Normal', 'Viral Pneumonia']):
    """
    :param path: path of dataset
    :param classes: folder names or labels
    :return: X,  the list of img paths
             y,  the list of img labels
    """
    X = []
    y = []
    for className in classes:
        for imgPath in glob.glob(os.path.join(path, className, '*')):
            # get the path of all files in path/className
            X.append(imgPath)
            y.append(CLASS_MAPPING[className])
    return np.asarray(X), np.asarray(y)


class CovidDataset(Dataset):
    def __init__(self, X, y, transform=transforms.ToTensor()):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img_path = self.X[index]
        # img = plt.imread(img_path)
        img = self.transform(Image.open(img_path).convert('L'))
        label = self.y[index]
        return img, torch.tensor(label)


def createDataLoader(X, y, batch_size=32, test_ratio=0.3, transform=transforms.ToTensor()):
    """
    :param X: The list of image paths
    :param y: The list of labels
    :param test_ratio: the proportion of test data
    :return: data loaders of training set and test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)
    train_dataset = CovidDataset(X_train, y_train, transform)
    test_dataset = CovidDataset(X_test, y_test, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader



if __name__ == '__main__':
    X, y = readData()
    print(f'X.shape {len(X)} - type(X) {type(X)}')
    print(f'y.shape {len(y)} - type(y) {type(y)}')

    dataset = CovidDataset(X, y)
    img = dataset[10]
    print(img)
    for i in range(len(dataset)):
        img, label = dataset[i]


