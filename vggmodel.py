import torch
import torch.nn.functional as F
import covidata


class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv9 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.conv10 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        self.conv11 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.conv12 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.conv13 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(4608, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 4)

    def forward(self, x):
        batch_size = x.size(0)
        # 1, 299, 299
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pooling(x)

        # 64, 147, 147
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pooling(x)

        # 128, 71, 71
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pooling(x)

        # 256, 32, 32
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pooling(x)

        # 512, 13, 13
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pooling(x)

        # 512, 3, 3
        x = x.view(batch_size, -1)

        # 4608
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


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

    model = VGG16()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X, y = covidata.readData()
    train_loader, test_loader = covidata.createDataLoader(X, y, 32, 0.2)  # batch_size = 16

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(15):
        train(model, epoch, train_loader, optimizer, criterion, device)
        test(model, test_loader, device)