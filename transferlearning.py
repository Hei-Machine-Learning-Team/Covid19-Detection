import torch
import torch.nn as nn
import torchvision
import covidata
import torchvision.transforms as transforms
from utils import train, test

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.vgg16(pretrained=True)
    # print(model.classifier[6].out_features)  # 1000

    # Freeze training for all layers
    for param in model.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    features.extend([
        nn.Linear(num_features, 64),
        nn.Linear(num_features, 3)])  # Add our layer with # features outputs
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
    X_test, y_test, train_loader, test_loader = covidata.createDataLoader(X, y, 32, 0.2, transform)

    for epoch in range(10):
        train(model, epoch, train_loader, optimizer, criterion, device)
        test(model, test_loader, device)