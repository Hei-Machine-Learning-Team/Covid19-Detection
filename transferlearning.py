import torch
import csv
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import covidata
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

from utils import train, test


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
                """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        # model_ft.fc = nn.Sequential(
        #     nn.Linear(2048, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 3))
        # input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg16":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

        # model_ft.classifier = nn.Sequential(nn.Linear(25088, 4096),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Linear(4096, 128),
        #                                     nn.ReLU(inplace=True),
        #                                     nn.Linear(128, num_classes))
        input_size = 224

    elif model_name == "vgg11":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_transforms(input_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transform, test_transform


def get_param_to_update(model_ft, feature_extract=True):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return params_to_update


if __name__ == '__main__':

    model_name = "alexnet"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, input_size = initialize_model(model_name, 3, False, False)
    model.to(device)

    # read data
    X, y = covidata.readData()
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
    # create dataset
    train_transform, test_transform = get_transforms(input_size)
    train_dataset = covidata.CovidDataset(X_train, y_train, train_transform)
    valid_dataset = covidata.CovidDataset(X_valid, y_valid, test_transform)
    # create dataloader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # get parameters
    params = get_param_to_update(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)

    # save test dataset for later evaluation
    np.save("Xtest.npy", X_test)
    np.save("ytest.npy", y_test)

    best_accuracy = 0.0
    loss_record = []
    accuracy_record = []
    for epoch in range(150):
        epoch_loss, losses = train(model, epoch, train_loader, optimizer, criterion, device)
        accuracy = test(model, valid_loader, device)
        # save the best model
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), "./save.pt")
        # loss_record += losses
        loss_record.append(epoch_loss)
        accuracy_record.append(accuracy)

        # record loss and accuracy every epoch
        with open('record.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i+1 for i in range(epoch+1)])
            writer.writerow(loss_record)
            writer.writerow(accuracy_record)
            f.close()

    # # save model and test set
    torch.save(model.state_dict(), "./save-end.pt")