# Detector de caras Falsas

## Índice
1. [Introducción](#introducción)
2. [Conjunto de datos](#conjunto-de-datos)
3. [Machine Learning](#machine-Learning)<br>
    3.1. [Histogram of oriented gradients](#HOG)<br>
    3.2. [Local binary patterns](#LBP)<br>
4. [Detección de caras](#detección-de-caras)
5. [Deep Learning](#deep-Learning)<br>
    5.1. [Facenet](#Facenet)<br>
    5.2. [Vggface](#vgg-face)<br>
    5.3. [Openface](#Openface)<br>
    5.4. [DeepFace](#DeepFace)<br>
8. [Trabajos Futuros](#trabajos-futuros)
7. [Bibliografía](#bibliografía)


## Introducción

El objetivo de este repositorio es explorar y comparar diferentes técnicas para la detección de caras falsas generadas por [modelos generativos](https://arxiv.org/abs/1406.2661). Para ello se han comparado diferentes clasificadores junto con distintivos esquemas de representación del *machine learning* clásico al *Deep Learning*.

Para todas las combinaciones y técnicas empleadas se ha elegido como método de evaluación un 5-flod mediante el método *StratifiedKFold* de *scikit-learn*. Como clasificadores se han empleado los siguientes:

* [Máquina de vectores soporte](https://en.wikipedia.org/wiki/Support-vector_machine): variando los parámetros de regularización y utilizando un kernel radial.
* [k nearest neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm): variando la k entre 5-40 vecinos y utilizando las distancia coseno y euclidea.
* [Random forest](https://en.wikipedia.org/wiki/Random_forest): variando el número de estimadores, la profundidad de los árboles, el tipo de división y de características.
* [Stacking de clasficadores](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/): se han agrupado varios tipos diferentes de clasificadores para generar un conjunto de datos basado en la puntación que generan esos modelos por cada muestra, para que después con una máquina de vectores soporte se pueda hacer una clasificación sobre ese conjunto de datos genereado.

Para el ajuste de los hiper-parámetros se han utilizado las funciones *GridSearchCV* y *RandomizedSearchCV*.Además, Para la evalución de los métodos empleamos se han eligo tres métricas básicas: *precision*, *recall* y *accuracy*


## Conjunto de datos

Se ha elegido una versión a menor resolución del conjunto de datos **Real and Fake Face Detection** disponible en [Kaggle](https://www.kaggle.com/ciplab/real-and-fake-face-detection) pasando de una resolución inicial de   600x600   a   un   conjunto   de   imágenes   160x160,   que   permite   acelerar   el   procesamiento. Este *dataset* está dividido en imágenes de caras reales(1081 muestras) y falsas(980 muestras), donde para las falsas existen tres categorías: sencilla(240 muestras), normal(480 muestras) y difícil(240 muestras).

## Machine Learning

Como técncicas de representación de imagenes clásicas se han utilizado [Histograma de gradientes orientados(HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) y [patrones binarios locales(LBP)](https://en.wikipedia.org/wiki/Local_binary_patterns), para ambos se ha transformado la imagen en un vector de caracterisitcas y se han aplicado estas técnicas:

* HOG: vectores de 32 dimensiones
* LBP: vectores de 531 dimensiones

Se representamos ambas técncias con un [t-SNE](https://lvdmaaten.github.io/tsne/) podemos apreciar que no generan un espacio de representación para las imagenes fácil de clasificar.

<p align="center">
  <img src="img/hog_rep.jpg" alt="vectores hog">
</p>
<p align="center">
  Figura 2: Diagrama UML
</p>
<br>

## HOG

Se ha empleado los clasificadores, junto con una variante del SVM utilizando como representación una versión compactada del HOG con un PCA conservando el 95% de variabilidad.

|                       | Precision(%) | Recall(%) | Acc(%) |
|-----------------------|:------------:|:---------:|:------:|
|  HOG + SVM (Baseline) |     **59.8**     |     53    |  56.2  |
| HOG + stacking(model) |    52,658    |   **62,292**  | 56,003 |
|        HOG + RF       |     56,16    |   49,271  | **58,061** |
|    HOG + PCA + SVM    |    54,274    |   60,729  | 57,474 |


## LBP

Se repiten los experimentos utilizando LBP.

|                       | Precision(%) | Recall(%) | Acc(%) |
|:---------------------:|:------------:|:---------:|:------:|
|       LBP + SVM       |     55,2     |   59,167  | 58,205 |
| LBP + stacking(model) |     57,04    |   58,95   |  59,67 |
|        LBP + RF       |    **60,645**    |   43,125  | **60,068** |
|    LBP + PCA + SVM    |     55,77    |     **61**    |   59   |

## Detección de caras

Antes de utilizar modelos basados en deep learning hemos elegido una región de interes en la imagen donde centrar los modelos, más concretamente se va a utilizar únicamnete la zone de la cara centrandose en la nariz, ojos y bocas de las muestras. Como detector de caras se ha empelado [retinaface](https://arxiv.org/abs/1905.00641), de media se consigue una región de interes de 114x118. Este detector funciona bien tanto para caras falsas como para reales.

<p align="center">
  <img src="/Database_real_and_fake_face_160x160/fake/easy_100_1111.jpg" alt="fake_face">
</p>
<p align="center">
  Figura 2: Diagrama UML
</p>
<br>