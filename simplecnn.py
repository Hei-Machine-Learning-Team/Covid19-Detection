import torch
import torch.nn.functional as F
import covidata
import torchvision.transforms as transforms
from utils import train, test, eval_model


class SimpleCNN(torch.nn.Module):
    """
    Simple Convolutional Neural Network
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=7)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=7)
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=30, kernel_size=7)
        self.pooling = torch.nn.MaxPool2d(3)
        self.fc1 = torch.nn.Linear(1920, 320)
        self.fc2 = torch.nn.Linear(320, 40)
        self.fc3 = torch.nn.Linear(40, 4)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x





if __name__ == '__main__':

    model = SimpleCNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X, y = covidata.readData()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    X_test, y_test, train_loader, test_loader = covidata.createDataLoader(X, y, 16, 0.2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(10):
        train(model, epoch, train_loader, optimizer, criterion, device)
        test(model, test_loader, device)