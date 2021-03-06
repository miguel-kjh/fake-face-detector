# Fake face detector
<!--Autor: Miguel Ángel Medina Ramírez

## Índice
1. [Introducción](#introducción)
2. [Conjunto de datos](#conjunto-de-datos)
3. [Machine Learning](#machine-Learning)<br>
    3.1. [Histogram of oriented gradients](#HOG)<br>
    3.2. [Local binary patterns](#LBP)<br>
4. [Detección de caras](#detección-de-caras)
5. [Deep Learning](#deep-Learning)<br>
6. [Score Stacking](#score-stacking)
7. [Estudio sobre las caras falsa](#estudio-sobre-caras-falsas)
8. [Trabajos Futuros](#trabajos-futuros)
9. [Conclusiones](#conclusiones)-->

## Introduction

The aim of this repository is to explore and compare different techniques for false face detection generated by [generative models](https://arxiv.org/abs/1406.2661). For this purpose, different classifiers have been compared together with distinctive representation schemes from classical *machine learning* to *Deep Learning*.

For all combinations and techniques used, a 5-flod by means of the *StratifiedKFold* method of *scikit-learn* has been chosen as evaluation method, furthermore a normalisation of the data set takes place in order to set the values between [0, 1]. The following classifiers have been used as classifiers:

* [Support Vector Machine](https://en.wikipedia.org/wiki/Support-vector_machine): varying the regularisation parameters and using a radial kernel.
* [k nearest neighbour](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm): varying the k between 5-40 neighbours and using the cosine and Euclidean distance.
* [Random forest](https://en.wikipedia.org/wiki/Random_forest): varying the number of estimators, tree depth, splitting and feature type.
* [Stacking of classifiers](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/): several different types of classifiers (logistic regression, KNN, a decision tree, SVM, Gaussian network, random forest) have been grouped together to generate a dataset based on the score generated by these models for each sample, so that a classification can then be made with a support vector machine on this generated dataset.

For the adjustment of the hyper-parameters we have used the functions *GridSearchCV* (for the SVM and for the KNN) and *RandomizedSearchCV* (for the RF) and three basic metrics have been chosen for the evaluation of the methods used: *precision*, *recall* and *accuracy*.

## Dataset

A lower resolution version of the **Real and Fake Face Detection** dataset available in [Kaggle](https://www.kaggle.com/ciplab/real-and-fake-face-detection) has been chosen, going from an initial resolution of 600x600 to a 160x160 image set, which allows to speed up the processing. This *dataset* is divided into real(1081 samples) and fake(980 samples) face images, where for fake faces there are three categories: simple(240 samples), normal(480 samples) and difficult(240 samples).

## Machine Learning

As classical image representation techniques, [Histogram of Oriented Gradients(HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) and [Local Binary Patterns(LBP)](https://en.wikipedia.org/wiki/Local_binary_patterns) have been used, for both of them the image has been transformed into a vector of characteristics and these techniques have been applied:
 
* HOG: 32-dimensional vectors
* LBP: 531-dimensional vectors
 
If we represent both techniques with a [t-SNE](https://lvdmaaten.github.io/tsne/) we can see that they do not generate a representation space for the images that is easy to classify.

<p align="center">
  <img src="/img/hog.png" alt="vectores hog">
  <img src="/img/lbp.png" alt="vectores lbp">
</p>
<p align="center">
  Figure 1: Representation of the descriptors obtained with HOG and LBP.
</p>
<br>

## HOG

The classifiers have been used, together with a variant of the SVM using a compact version of the HOG as a proxy with a PCA preserving 95% variability.

|                       | Precision(%) | Recall(%) | Acc(%) |
|-----------------------|:------------:|:---------:|:------:|
|  HOG + SVM (Baseline) |     **59.8**     |     53    |  56.2  |
| HOG + stacking(model) |    52,658    |   **62,292**  | 56,003 |
|        HOG + RF       |     56,16    |   49,271  | **58,061** |
|    HOG + PCA + SVM    |    54,274    |   60,729  | 57,474 |


## LBP

Experiments are repeated using LBP.

|                       | Precision(%) | Recall(%) | Acc(%) |
|:---------------------:|:------------:|:---------:|:------:|
|       LBP + SVM       |     55,2     |   59,167  | 58,205 |
| LBP + stacking(model) |     57,04    |   58,95   |  59,67 |
|        LBP + RF       |    **60,645**    |   43,125  | **60,068** |
|    LBP + PCA + SVM    |     55,77    |     **61**    |   59   |

## Detección de caras

Before using models based on deep learning we have chosen a region of interest in the image where to focus the models, more specifically we are going to use only the area of the face focusing on the nose, eyes and mouths of the samples. As a face detector we have used [retinaface](https://arxiv.org/abs/1905.00641), on average a region of interest of 114x118 is achieved. This detector works well for both fake and real faces.

<p align="center">
  <img src="/Database_real_and_fake_face_160x160/fake/hard_100_1111.jpg" alt="fake_face_total">
  <img src="/Database_real_and_fake_only_face/fake/hard_100_1111.jpg" alt="fake_face">
</p>
<p align="center">
  Figure 2: Example of cutting out false faces
</p>
<br>

<p align="center">
  <img src="/Database_real_and_fake_face_160x160/real/real_00002.jpg" alt="fake_face_total">
  <img src="/Database_real_and_fake_only_face/real/real_00002.jpg" alt="fake_face">
</p>
<p align="center">
  Figure 3: Example of face cropping
</p>
<br>

For each face a [normalisation L2](https://paulrohan.medium.com/euclidean-distance-and-normalization-of-a-vector-76f7a97abd9) is produced before using some of the models proposed.


## Deep Learning

Four types of deep networks have been chosen for the transformation of faces into embeddings for classification.

* [Facenet](https://arxiv.org/abs/1503.03832): Generates a 128-dimensional representation space.
* [VGG-face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/): Generates a 2622-dimensional representation space.
* [Openface](https://cmusatyalab.github.io/openface/): Generates a 128-dimensional representation space.
* [deepface](https://ieeexplore.ieee.org/document/6909616): Generates a 4096-dimensional representation space.

As the VGG-face and Deepface generate high dimensionality descriptors, they have been compressed with a PCA preserving 95% of variability, in this way the vector space generated by VGG-face goes from 2622 components to 302 and the 4096 components of Deepface go to 641. To work with them, the framework [DeepFace](https://pypi.org/project/deepface/) has been used.


|                                 | Precision(%) | Recall(%) | Acc(%) |
|:-------------------------------:|:------------:|:---------:|:------:|
|       ROI + FaceNet + SVM       |      **57**      |   58,438  | 59,187 |
|       ROI + FaceNet + KNN       |      53      |   40,729  | 54,778 |
|        ROI + FaceNet + RF       |      56      |   39,375  |  57,1  |
| ROI + FaceNet + stacking(model) |    54,164    |   61,25   | 57,324 |
|       ROI + Openface + SVM      |    54,621    |   **61,771**  | 57,815 |
|  ROI + VGGface + PCA(95%) + SVM |    54,305    |   53,854  | 56,885 |
| ROI + Deepface + PCA(95%) + SVM |    56,623    |   59,792  | **59,481** |

## Score Stacking

One idea to improve the efficiency of the methods proposed is to perform a fusion at the score level, similar to a boosting technique but using the results of the classifiers to build a dataset to perform another classification, an example of this strategy is shown in the figure. In this case we try to use a fusion between machine learning methods and deep learning methods based on SVM.

<p align="center">
  <img src="/img/score_stacking.png" alt="score stacking">
</p>
<p align="center">
  Figure 4: Example of score fusion for gender classification
</p>
<br>

Two experiments are carried out:

1. Using the results obtained with the face representation models(Vggface, deepface, openface and facenet) with the SVM classifier.

2. SVMs trained using HOG and LBP as representation space.

For both obtained datasets an SVM is used for classification.


|                         | Precision(%) | Recall(%) | Acc(%) |
|:-----------------------:|:------------:|:---------:|:------:|
| DL  Stacking(SVM) + SVM |    **69,342**    |   **69,062**  | **70,994** |
| ML  Stacking(SVM) + SVM |    57,651    |   57,292  | 59,972 |

## Research on false faces

In this section we have compared the best technique used, which would be score fusion with *Deep Learnig*, training with the simple, normal and difficult samples of the set of fake faces, to balance the experiments we have used an equal number of samples of real faces randomly chosen for each level of difficulty.

| DL Stacking(SVM) + SVM | Precision(%) | Recall(%) | Acc(%) |
|:----------------------:|:------------:|:---------:|:------:|
|        Easy(240 muestras)       |    66,514    |    67,5   | 66,667 |
|        Mid(480 muestras)        |    **76,026**    |   **73,75**   | **75,104** |
|        Hard(240 muestras)       |    72,182    |    72,5   | 72,083 |

We observe that the best ensemble is the normal ensemble, followed by the difficult and the simple ensemble. This may be due to the fact that the normal one has more samples than the other two and the methods based on face recognition with deep networks are trained mainly with well-formed faces, which is the opposite of what is found in the simple set, which is why it is the worst of the three.

## Future work

As possible future work, in view of the results obtained with the techniques used, it would be of interest to
 
1. The re-training of some of the *Deep Learning* models used, although we only have 2061 samples, it would be interesting to evaluate the impact of partially re-training the weights of the Vggface or the facenet to achieve a better representation and re-apply the same experiments.
 
2. Using [Adaptive Manipulation Trace Extraction Networks (AMTEN)](https://arxiv.org/abs/2005.04945), have recently become state of the art in the detection of manipulated images. For fake face detection CNNs only learn representations of image content, however, using a pre-processing AMTEN allows transforming an image by removing its content and focusing on highlighting the manipulation traces making subsequent CNNs generate better descriptors.


## Conclusions

In conclusion, as can be seen from representing the descriptors generated with the various techniques, this dataset and this problem in general is quite complex to solve simply with methods classified from *Machine Learning* or *Deep Learning*. In order to reach 70% *accuracy* a boosting technique based on *score* fusion had to be devised, which shows how useful these techniques are and how well they perform in achieving a representation space for better partitioning or classification.
 
In the case of false faces or *Deepfake* they are a rather complex problem if one does not know the generative models that have been used to generate them or the arbitrator models that have been used to train the *GANs*, it is evident that achieving a better generation of descriptors helps in the detection of false faces.
