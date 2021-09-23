import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train(model, epoch, train_loader, optimizer, criterion, device):
    model.train()  # set model to training mode
    epoch_loss = 0.0
    losses = []
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        if batch_idx % 50 == 49:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss))
            losses.append(running_loss)
            running_loss = 0.0

    return epoch_loss/len(train_loader.dataset), losses


def test(model, test_loader, device):
    model.eval()  # Set model to evaluate mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))
    return accuracy


def eval_model(model, test_loader):
    # return a confusion matrix to evaluate the model
    # confusion_matrix = torch.zeros(size=(4, 4), dtype=int)
    # with torch.no_grad():
    #     for data in test_loader:
    #         inputs, target = data
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, dim=1)
    #         for k in range(len(target)):
    #             i = target[k]
    #             j = predicted[k]
    #             confusion_matrix[i][j] += 1
    ret_matrix = np.zeros(shape=(3, 3))
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            batch_matrix = confusion_matrix(target, predicted, labels=[0, 1, 2])
            ret_matrix += batch_matrix
    return ret_matrix


def draw_confusion_matrix(matrix):
    f, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax, cmap="Blues")

    ax.set_title('Confusion Matrix')  # title
    ax.set_xlabel('Predict')  # x
    ax.set_ylabel('True')  # y


def get_stat(matrix, label):
    idx = [0, 1, 2]
    idx.remove(label)
    TP = matrix[label][label]  # True Positive
    FN = np.sum([matrix[label][i] for i in idx])  # False Negative
    FP = np.sum([matrix[i][label] for i in idx])  # False Positive
    TN = np.sum([matrix[i][j] for i in idx for j in idx])  # True Negative
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2*TP / (2*TP + FP + FN)
    print("label: ", label)
    print("Precision: ", Precision)
    print("Recall: ", Recall)



