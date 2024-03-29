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

`1st Evaluation comments`

* Knowledge of existing models need to be improved. 
* Research problem needs to be highlighted.
* More comparative parameters must be included (AUC, ROC) , etc.
* K-fold cross-validation must be carried out. 
* Computation time must be reported.

`2nd Evaluation comments`

* (Preprocessing) Clean noise from the dataset. apply foreground/backgroud noise removing technique
* Keep the image size 256x256
* just for testing purpose(reduce the images and train the model) 200 for training 50 for testing
* SVM and K-NN comparision
* Add CNN Layers 2,3,4,5 just for testing purpose
* What samples are put in PCR? Swab test or blood samples.
* Convolution layer depth
* cite paper from which the dataset is taken.
* ROC needs to be checked
* do transfer learning on Alexnet, ResNet, DenseNet, VGG-net
* Dont compare accuracy and loss with other papers.
* before inserting the data into CNN what operations They are doing? get some idea from their codes
* Apply Wavelet Scattering Transform in preprocessing
* dwt, swt wavelets do spartiaty reduce.
* Testing time should be reported
* Approximation image (type of wavelet scattering transform) recommended
* PCA usage


`Meeting updates 05 March, 2021∶`

* Create a survey of all available chest xray datasets, with citations, using tabular format. 
* Increase the train and test dataset to create realistic testing conditions to increase the robustness. 
* Data augmentation can be used if needed for creating a robust model. 
* Contrast with the most recent paper results. 
* Improve the sensitivity, specificity, accuracy, percision.
* wrtiing thesis and jurnal paper. 

________________________________________________________________________________________________________________________

#### Figure1-COVID-chestxray-dataset https://github.com/agchung/Figure1-COVID-chestxray-dataset
|Covid19| 
|:-----:|
|  55   |
> - Covid-net

#### Actualmed-COVID-chestxray-dataset https://github.com/agchung/Actualmed-COVID-chestxray-dataset
|  Covid19 |  No Finding |  Not Defined |  Total | 
|:--------:|:-----------:|:------------:|:------:|
| 58       | 127         | 53           | 238    |
> - Covid-net


#### COVID-19 Radiography Database https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
|  Covid19 |  Viral Pneumonia |  Normal |  Total | 
|:--------:|:----------------:|:-------:|:------:|
| 1200     |1345              | 1341    | 3886   |
> - Covid-net<br>
> - A Deep Learning Approach for COVID-19 & Viral Pneumonia Screening with X-ray Images (December 2020) <br>

#### covid-chestxray-dataset https://github.com/ieee8023/covid-chestxray-dataset
|  Pneumonia/Viral/COVID-19 |  Pneumonia|  No Finding | Total | 
|:-------------------------:|:---------:|:-----------:|:-----:|
| 520                       | 81        | 18          | 619   |
> - Covid-net <br>
> - Deep-COVID: Predicting COVID-19 from chest X-ray images using deep transfer learning <br>
> - Deep Learning COVID-19 Features on CXR using Limited Training Data Sets <br>
> - COVID-19 Deep Learning Prediction Model Using Publicly Available Radiologist-Adjudicated Chest X-Ray Images as Training Data: Preliminary Findings(18 August 2020)<br>
> - Automated detection of COVID-19 cases using deep neural networks with X-ray images <br>

#### rsna-pneumonia-detection-challenge https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
|  Lung Opacity |   No Lung Opacity / Not Normal |   Normal |   Total | 
|:-------------:|:------------------------------:|:--------:|:-------:|
| 9555          | 11821                          | 8851     | 30227   |
> - Covid-net


#### CoronaHack -Chest X-Ray-Dataset https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset
|  Label  |   Label_1_Virus_category |  Label_2_Virus_category |   Total | 
|:-------:|:------------------------:|:-----------------------:|:-------:|
| Normal  |                          |                         | 1576    |
| Pnemonia| Stress-Smoking           | ARDS                    | 2       |
| Pnemonia| Virus                    |                         | 1493    |
| Pnemonia| Virus                    | COVID-19                | 58      |
| Pnemonia| Virus                    | SARS                    | 4       |
| Pnemonia| bacteria                 |                         | 2772    |
| Pnemonia| bacteria                 | Streptococcus           | 5       |
|         |                          |      **Total**          |**5910** |
 
> - Deep Learning COVID-19 Features on CXR using Limited Training Data Sets


#### COVID-19 Patients Lungs X Ray Images 10000 https://www.kaggle.com/nabeelsajid917/covid-19-x-ray-10000-images
|  Covid |  Normal |  Total | 
|:------:|:-------:|:------:|
| 70     | 28      | 98     |

> - A Deep Learning Approach for COVID-19 & Viral Pneumonia Screening with X-ray Images (December 2020)
________________________________________________________________________________________________________________________


# Performance Metrics of Binary Class Model

![Table](https://github.com/source-droid/Machine-Learning-based-Automatic-Covid-19-detection-using-Lung-s-Scans/blob/main/Docs/Performance_Metrics_of_binary_classification_model.png)


# Performance Metrics of Three Class Model

|       |   Accuracy|    Specificity|    Sensitivity(recall)|    Precision|    F1_score|
|-------|:---------:|:-------------:|:---------------------:|:-----------:|:----------:|
|Overall|    97.2061|       97.8898 |               95.8668 |     95.8668 |    95.8668 |
|Covid-19|    97.2061|       99.4845 |               96.997  |     98.1763 |    97.5831 |
|Normal|    97.2061|       98.6486 |               96.6543 |     94.5455 |    95.5882 |
|Pneumonia|    97.2061|       98.6388 |               93.6803 |     94.382  |    94.0299 |



# Three Class Model (Covid-19-Normal-Pneumonia)

## Dataset Details

|           |Covid |Normal |Pneumonia|Total     |
|:---------:|:---: |:-----:|:-------:|:--------:|
|**Train**  |1330  |1072   |1076     |3478      |
|**Test**   | 333  | 269   |269      |871       |
|**Total**  |1663  |1341   |1345     |**4349**  |


`Training Time In Google Colab   1h 4min 14s`


# Classification Report

|            |precision|    recall|  f1-score|   support|
|:----------:|:-------:|:--------:|:--------:|:--------:|
|        0.0 |     0.98|     0.97 |    0.98  |    333   |
|         1.0|     0.95|      0.97|     0.96 |     269  |
|        2.0 |     0.94|     0.94 |    0.94  |    269   |
|            |         |          |          |          |
|   accuracy |         |          |    0.96  |    871   |
|  macro avg |     0.96|     0.96 |    0.96  |    871   |
|weighted avg|     0.96|      0.96|     0.96 |     871  |

covid19 = 0, normal = 1 , Pnuemonia = 2<br>
Accuracy = 96.1%


# Accuracy and Loss Plot
![acc_loss_plot](https://github.com/source-droid/Machine-Learning-based-Automatic-Covid-19-detection-using-Lung-s-Scans/blob/main/Docs/Plot%20C+N+P%20With%20Droupout%20epoch20.png)


# Confusion Matrix
![confusion1](https://github.com/source-droid/Machine-Learning-based-Automatic-Covid-19-detection-using-Lung-s-Scans/blob/main/Docs/confusion%20matrix%20C+N+P%20With%20Droupout%20epoch20.png)

________________________________________________________________________________________________________________________________________________________________________


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
