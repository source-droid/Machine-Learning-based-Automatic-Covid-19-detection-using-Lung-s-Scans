# Machine Learning based Automatic Covid-19 detection using Lung’s Scans

# Team Members:
**Syed Aakash Ali** <br>
Student, Department of Electrical Engineering<br>
Sukkur IBA University<br>
Sukkur, Sindh, Pakistan<br>
Email: syedaakash.be17@iba-suk.edu.pk<br>

**Hassan Azam**<br>
Student, Department of Electrical Engineering<br>
Sukkur IBA University<br>
Sukkur, Sindh, Pakistan<br>
Email: hassanazam.be17@iba-suk.edu.pk<br>

# Supervisor:
**Dr. Safeer Hyder** <br>
Associate Professor, Department of Electrical Engineering<br>
Sukkur IBA University<br>
Sukkur, Sindh, Pakistan<br>
Email: safeer.hyder@iba-suk.edu.pk<br>

`Evaluator's comments`

* Knowledge of existing models need to be improved. 
* Research problem needs to be highlighted.
* More comparative parameters must be included (AUC, ROC) , etc.
* K-fold cross-validation must be carried out. 
* Computation time must be reported.



`Meeting updates 05 March, 2021∶`

* Create a survey of all available chest xray datasets, with citations, using tabular format. 
* Increase the train and test dataset to create realistic testing conditions to increase the robustness. 
* Data augmentation can be used if needed for creating a robust model. 
* Contrast with the most recent paper results. 
* Improve the sensitivity, specificity, accuracy, percision.
* wrtiing thesis and jurnal paper. 

# Brief Project Introduction
The coronavirus (COVID-19) pandemic is putting healthcare systems across the world under unprecedented and increasing pressure according to the World Health Organization (WHO). With the advances in computer algorithms and especially Artificial Intelligence, the detection of this type of virus in the early stages will help in fast recovery and help in releasing the pressure off healthcare systems. One of the crucial steps in fighting COVID-19 is the ability to detect infected patients early enough and put them under special care. Detecting this disease from radiography and radiology images is perhaps one of the fastest ways to diagnose the patients. <br>

Some of the early studies showed specific abnormalities in the chest radiograms of patients infected with COVID-19. Inspired by earlier works, we study the application of deep learning models to detect COVID-19 patients from their chest radiography images. We aim to present the use of deep learning for the high-accuracy detection of COVID-19 using chest X-ray images. Publicly available X-ray images of Covid-19 infected patients along with healthy radiograms will be used for the training and validation of the classifier. The feature-based classifier will be developed to discriminate and classify COVID-19 patients. Further, cross-validation will also be conducted on pneumonia radiograms to check the robustness of the deep learning algorithm.<br>

This is encouraging, as it shows the promise of using X-ray images for COVID-19 diagnostics. This study will be conducted on a set of publicly available images. The comparison will be carried out with Neural Networks and results will be evaluated in terms of classification accuracy and precision and recall based AUC and ROC curves.


# Project Workflow
![workflow](https://github.com/source-droid/Machine-Learning-based-Automatic-Covid-19-detection-using-Lung-s-Scans/blob/main/Project%20Workflow.PNG)

Task 1: Getting Started with Google Colab Environment/ jupyter notebook/Kaggle & importing necessary libraries

Task 2: Importing, Cloning & Exploring Dataset

Task 3: Data visualization (Image Visualization)

Task 4: Data augmentation & Normalization

Task 5: Building Convolutional neural network model

Task 6: Compiling & Training CNN Model

Task 7: Performance evaluation & Testing the model & saving the model for future use

# Expected Outcome
Our model should be able to classify the lung’s scans correctly whether it is COVID +ve or –ve.

# Dataset


Two sources are used to create this dataset:
* [Covid-Chestxray-Dataset](https://github.com/ieee8023/covid-chestxray-dataset), for COVID-19 X-ray samples only
* [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database), for Non-COVID samples and COVID samples



1. [Covid Chest X Ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset)

520 Covid-19 images are taken from above dataset

2. [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

1341 normal and 1143 Covid-19 images are taken from above dataset


# `Our Dataset`
Split | Normal | Covid19|Total|
------|:------:|:------:|:---:|
Train | 1072   |1330    |1922 |
Test  | 269    |333     |602  |
**Total**| **1341**   | **1663**    |   **3004**   |

Some of the sample images from dataset are shown below.

![samples](https://github.com/shervinmin/DeepCovid/blob/master/results/covid5k_samples.png)

## Credits and Links
**Dataset:** [Covid Chest X Ray Dataset](https://github.com/ieee8023/covid-chestxray-dataset)<br>
**Dataset:** [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

**Paper:**
[Deep-COVID: Predicting COVID-19 From Chest X-Ray Images Using Deep Transfer Learning](https://arxiv.org/pdf/2004.09363.pdf)
