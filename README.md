# Detector de caras Falsas

## Índice
1. [Introducción](#introducción)
2. [Conjunto de datos](#conjunto-de-datos)
3. [Machine Learning](#machine-Learning)<br>
    3.1. [Histogram of oriented gradients](HOG)<br>
    3.2. [Local binary patterns](LBP)<br>
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


## Conjunto de datos

Se ha elegido una versión a menor resolución del conjunto de datos **Real and Fake Face Detection** disponible en [Kaggle](https://www.kaggle.com/ciplab/real-and-fake-face-detection) pasando de una resolución inicial de   600x600   a   un   conjunto   de   imágenes   160x160,   que   permite   acelerar   el   procesamiento. Este *dataset* está dividido en imágenes de caras reales(1081 muestras) y falsas(980 muestras), donde para las falsas existen tres categorías: sencilla(240 muestras), normal(480 muestras) y difícil(240 muestras).