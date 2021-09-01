import os
import glob
import numpy as np


def readData(path="./COVID-19_Radiography_Dataset/", classes=['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']):
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
            y.append(className)
    return np.asarray(X), np.asarray(y)


if __name__ == '__main__':
    X, y = readData()
    print(f'X.shape {len(X)} - type(X) {type(X)}')
    print(f'y.shape {len(y)} - type(y) {type(y)}')

