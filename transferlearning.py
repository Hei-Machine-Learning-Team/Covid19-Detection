import torch
import torch.nn as nn
import torchvision
import covidata
import torchvision.transforms as transforms


def train(model, epoch, train_loader, optimizer, criterion, device):
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
        if batch_idx % 50 == 49:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss))
            running_loss = 0.0


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
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the pretrained model from pytorch
    # vgg16 = models.vgg16_bn()
    # vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
    model = torchvision.models.vgg16(pretrained=True)
    # print(model.classifier[6].out_features)  # 1000

    # Freeze training for all layers
    for param in model.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 3)])  # Add our layer with # features outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.to(device)

    X, y = covidata.readData()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15, resample=False, fillcolor=0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_loader, test_loader = covidata.createDataLoader(X, y, 32, 0.2, transform)

    for epoch in range(10):
        train(model, epoch, train_loader, optimizer, criterion, device)
        test(model, test_loader, device)