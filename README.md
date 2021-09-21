# Covid19-Detection
The dataset we use in this project is [COVID-19 Radiography Database](kaggle.com/tawsifurrahman/covid19-radiography-database).  
Download the dataset, unzip and put it under /input. You can specify the dataset path in func covidata.py/readData.
## Structure of this project
- covidata.py: Read data and crate datasets and dataloaders
- evaluation.py: Script for trained model evaluation
- simplecnn.py: A simple CNN model
- transferlearning.py: Major code for model creation, transforms and set parameters to update
- utils.py: func train, test etc.
- vggmodels.py: VGG11 and VGG16 implementation
## How to start training
specify the `model_name` in `transferlearning.py`
To train the network, run  
```
python transferlearning.py
```
These 4 files will be automatically saved during the training process:
- saved.pt: the best model
- record.csv: training loss and accuracy on test set of every epoch
- Xtest.npy and ytest.npy: test dataset