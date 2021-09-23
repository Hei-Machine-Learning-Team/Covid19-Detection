import torch
import numpy as np
from covidata import createTestLoader
from utils import eval_model, get_stat
from transferlearning import initialize_model, get_transforms

if __name__ == '__main__':
    folder = "../savedmodels/alexnet/"
    model_path = folder + "save.pt"
    model_name = "alexnet"
    model, input_size = initialize_model(model_name, 3, False, False)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, test_transform = get_transforms(input_size)

    # create dataloader for test set
    X_test = np.load(folder + "Xtest.npy")
    y_test = np.load(folder + "ytest.npy")
    test_loader = createTestLoader(X_test, y_test, 32, test_transform)
    confusion_matrix = eval_model(model, test_loader)
    print("confusion matrix:")
    print(confusion_matrix)
    for i in range(3):
        get_stat(confusion_matrix, i)