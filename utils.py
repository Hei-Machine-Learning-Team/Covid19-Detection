import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train(model, epoch, train_loader, optimizer, criterion, device, is_inception=False):
    model.train()  # set model to training mode
    epoch_loss = 0.0
    losses = []
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward
        if not is_inception:
            outputs = model(inputs)
            loss = criterion(outputs, target.long())
        else:  # inception-v3 network
            outputs, aux_outputs = model(inputs)
            loss1 = criterion(outputs, target)
            loss2 = criterion(aux_outputs, target)
            loss = loss1 + 0.4 * loss2

        # backward + update
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        if batch_idx % 50 == 49:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss))
            losses.append(running_loss)
            running_loss = 0.0

    return epoch_loss/len(train_loader.dataset), losses


def test(model, val_loader, device):
    # test on validation data
    model.eval()  # Set model to evaluate mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
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


def draw_confusion_matrix(matrix, title="Confusion Matrix"):
    f, ax = plt.subplots()
    labels = ['COVID', 'Normal', 'Viral Pneumonia']
    sns.heatmap(matrix, annot=True, ax=ax, cmap="Blues", cbar=False, fmt='g', xticklabels=labels, yticklabels=labels)

    ax.set_title(title)  # title
    ax.set_xlabel('Predict')  # x
    ax.set_ylabel('True')  # y
    plt.show()


def get_stat(matrix, label):
    idx = [0, 1, 2]
    idx.remove(label)
    TP = matrix[label][label]  # True Positive
    FN = np.sum([matrix[label][i] for i in idx])  # False Negative
    FP = np.sum([matrix[i][label] for i in idx])  # False Positive
    TN = np.sum([matrix[i][j] for i in idx for j in idx])  # True Negative
    Accuracy = np.sum([matrix[i][i] for i in range(3)]) / np.sum(matrix)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Specificity = TN / (FP + TN)
    F1 = 2*TP / (2*TP + FP + FN)

    print("label: ", label)
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("Specificity: ", Specificity)
    print("F1: ", F1)
    print("Overall Accuracy: ", Accuracy)
    print("& %0.3f & %0.3f & %0.3f & %0.3f " % (Precision, Recall, Specificity, F1))




