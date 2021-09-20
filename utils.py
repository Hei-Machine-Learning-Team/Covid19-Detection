import torch


def train(model, epoch, train_loader, optimizer, criterion, device):
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

    return epoch_loss, losses


def test(model, test_loader, device):
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
    accuracy = 100 * correct / total, correct, total
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))
    return accuracy


def eval_model(model, test_loader):
    # return a confusion matrix to evaluate the model
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
    confusion_matrix = torch.zeros(size=(4, 4), dtype=int)
    for k in range(len(target)):
        i = target[k]
        j = predicted[k]
        confusion_matrix[i][j] += 1
    return confusion_matrix



