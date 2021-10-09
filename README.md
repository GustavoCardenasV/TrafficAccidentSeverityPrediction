# Accident Severity Prediction

## Project Overview:
The objective of this project is to create a model to predict a traffic crash severity type, ranging from "not severe accident" to "severe/fatal accident". The model aims to predict the type of severity considering a range of variables used as imputs and producing a binary output guessing between "not severe" or "severe/fatal". 

### DataSet
The original dataset used was obtained from the Police-Reported Crashes available on the Insurance Corporation of British Columbia (ICBC) website available in this link.

The data contains all the records of Trafic Crash (TC) reported by the police or submitted by individuals to police. The dataset consists on a total of a total of 93,546 observations containing a range of 42 different attributes such as "Month", "Weather", "Land Lse", "Road Surface" and one target variable named "Crash Type". 

### Evaluation
Regular plain text

## Project Set up:
The entirety of the this project can be run directly from the free Google platform "Google Colab". It is necessary to count with a Google account in order to have acces to the platform for this work. This project can be downloaded and run modifying the path on the variable __sys.path.append('select/the/path/here')__ located in the "Import the necessary libraries" section on top of the notebook. There is also a "file" variable __file = 'select/the/path/here/VPD_lowerMainLand.csv'__.

It is also necessary to use a program to compress and decompress files, such as 7zip or WinRar, in order to have acces to the saved models. The models can be find inside the "models" folder, compressed in a .7z format.

## Contents:

### Notebooks
The notebooks are located in the main section, there are 4 principal notebooks: "dataImportExploration.ipynb", "NaiveBayesClass_MixNaiveBayesClass.ipynb", "RandomForestClassifier.ipynb" and "MultyLayerPerceptron_NeuralNetwork.ipynb". Inside this notebooks can be find the walkthrough of the different Machine Learning (ML) algorithms and the Multy Layer Perceptron Neural Network (MLP NN)

### Modules
Inside the modules folder is located the "helperFunctions.py" file. This piece of code includes the methods definitios for the helper functions used across the different notebooks, helping with tasks such as "readFile" or "checkMissingValues". 

### Models
The models section contains the compressed folder containing the modeling for the ML models used in this work. In order to have acces to them, it is neecessary to make use of some program like 7zip or WinRar, both free. The files generated from the runs are saved in a .pkl (Pickle) file. The size of the compressed folder is 147MB and the size for the uncompressed folder is 1.57GB, just to take into consideration for storage pruposes.

### Data
The VPD_lowerMainLand.csv file can be finnd inside the "data" folder. This is the file where the "raw" information is contained, the 93,546 observations are limmited to the events reported by the police or submitted by individuals to police for the "Lower Mainland" area of Vancouver, British Columbia. The information listed is for the period of 2015 to 2019.

## Next Steps:
- Deeper data engineering preprocess
- Further modeling (i.e. Support Vector Machine)
- Develop the web GUI 

