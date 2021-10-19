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

## Result
### AlexNet

|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|681|42|0|
|Normal|44|2005|2|
|Viral Pneumonia|2|35|220|

### VGG11-BN
|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|713|12|2|
|Normal|61|1970|7|
|Viral Pneumonia|4|17|245|

### VGG16
|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|670|58|1|
|Normal|6|2042|6|
|Viral Pneumonia|2|19|227|

### InceptionV3
|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|679|57|2|
|Normal|17|1996|2|
|Viral Pneumonia|7|59|212|

### ResNet18
|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|667|63|1|
|Normal|11|2026|5|
|Viral Pneumonia|2|33|223|

### ResNet50
|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|673|62|1|
|Normal|9|2016|4|
|Viral Pneumonia|6|38|222|

### VGG16-Transfer Learning
|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|597|106|0|
|Normal|14|2036|7|
|Viral Pneumonia|0|13|258|

### ResNet50-Transfer Learning
|  |COVID|Normal|Viral Pneumonia|
|--|--|--|--|
|COVID|556|147|4|
|Normal|67|1976|5|
|Viral Pneumonia|9|52|215|