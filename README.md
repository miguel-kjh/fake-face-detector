# Detector de caras Falsas

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
9. [Conclusiones](#sonclusiones)
10. [Bibliografía](#bibliografía)


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
  <img src="/Database_real_and_fake_face_160x160/fake/hard_100_1111.jpg" alt="fake_face_total">
  <img src="/Database_real_and_fake_only_face/fake/hard_100_1111.jpg" alt="fake_face">
</p>
<p align="center">
  Figura 2: Ejemplo de recorte de caras falsas
</p>
<br>

<p align="center">
  <img src="/Database_real_and_fake_face_160x160/real/real_00002.jpg" alt="fake_face_total">
  <img src="/Database_real_and_fake_only_face/real/real_00002.jpg" alt="fake_face">
</p>
<p align="center">
  Figura 3: Ejemplo de recorte de caras
</p>
<br>

## Deep Learning

Se han elegido cuatro tipos de redes profundas para la transformación de las caras en embeddings para su clasificación.

* [Facenet](https://arxiv.org/abs/1503.03832): Genera una espacio de representación de 128 dimensiones.
* [VGG-face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/): Genera una espacio de representación de 2622 dimensiones.
* [Openface](https://cmusatyalab.github.io/openface/): Genera una espacio de representación de 128 dimensiones.
* [deepface](https://ieeexplore.ieee.org/document/6909616): Genera una espacio de representación de 4096 dimensiones.

Como la VGG-face y Deepface, generan despcritores de una alta dimensionalidad se han comprimido con un PCA consevando un 95% de variabilidad, de esta forma el espacio vectorial que genera VGG-face pasa de 2622 componentes a 302 y las 4096 componenetes de Deepface pasan a 641. Para trabajar con ellos se ha utilizado el framework [DeepFace](https://pypi.org/project/deepface/).

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

Una idea para mejorar la eficacia de los métodos planteados es realizar una fusión de a nivel de score, similar a una técnica de boosting pero utilizando los resultados de los clasificadores para construir un conjunto de datos donde realizar otra clasificación, en la figura  se veo un ejemplo de este estratetgía. En este caso probamos utilizar una fusión entre métodos del aprendizaje automático y los métodos del aprendijza profundo basada en SVM.

<p align="center">
  <img src="/img/score_stacking.png" alt="score stacking">
</p>
<p align="center">
  Figura 3: Ejemplo de fusión de score con clasificación de sexos
</p>
<br>

Se realizan dos experimentos:

1. Utilizando los resultados obtenidos con los modelos de representación de caras(Vggface, deepface, openface y facenet) con el clasificador SVM.

2. Los SVM entrenados utilziando como espacio de representación HOG y LBP.

Para ambos conjuntos de datos obtenidos se utiliza un SVM para la clasificación.

|                         | Precision(%) | Recall(%) | Acc(%) |
|:-----------------------:|:------------:|:---------:|:------:|
| DL  Stacking(SVM) + SVM |    **69,342**    |   **69,062**  | **70,994** |
| ML  Stacking(SVM) + SVM |    57,651    |   57,292  | 59,972 |


## Estudio sobre caras falsas

En este apartado se ha comparado la mejor técnica empleada, que sería la fusión de score con *Deep Learnig*, entrenando con las muestras sencillas, normales y difíciles de conjunto de caras falsas, para equilibrar los experimentos se han utilizando un número igual de muestras de caras reales elegidas aleatoriamente por cada nivel de dificultad.

| DL Stacking(SVM) + SVM | Precision(%) | Recall(%) | Acc(%) |
|:----------------------:|:------------:|:---------:|:------:|
|        Easy(240 muestras)       |    66,514    |    67,5   | 66,667 |
|        Mid(480 muestras)        |    **76,026**    |   **73,75**   | **75,104** |
|        Hard(240 muestras)       |    72,182    |    72,5   | 72,083 |

Observamos que el mejor conjunto es el de nivel normal, seguido de el difícil y del sencillo. Esto puede ser debido a que el normal dispone de más muestras que los otros dos y los métodos basados en reconocimiento de caras con redes profundas estas entrenados sobre todo con caras bien formadas todo lo contrario a lo que se encuentra en en conjunto sencillo, por ello es el peor de los tres.

## Trabajos Futuros

Como posibles trabajos futuros, viendo los resultados obtenidos con las técnicas empleadas sería interesante:
 
1. El re-entrenamiento de algunos de los modelos de *Deep Learning* empleados, aunque tan solo tengamos 2061 muestras sería interesante evaluar el impacto que tendría re-entrenar parcialmente los pesos de la Vggface o la facenet para lograr una mejor representación y volver a aplicar los mismos experimentos.
 
2. Utilizar [redes adaptativas de extracción de rastros de manipulación(AMTEN)](https://arxiv.org/abs/2005.04945), recientemente se han convertido en el estado del arte en la detección de imágenes manipuladas. Para la detección de caras falsas las CNN solamente aprenden representaciones del contenido de las imágenes, sin embargo, utilizar una AMTEN de preprocesamiento permite transformar una imagen eliminando su contenido y centrándose en resaltar los rastros de manipulación haciendo que las CNN posteriores generen mejores descriptores.

## Conclusiones

En suma, como se puede ver representando los descriptores generados con las diversas técnicas este conjunto de datos y este problema en general es bastante complejo para resolverlo simplemente con métodos clasificados desde el *Machine Learning* o el *Deep Learning*. Para llegar al 70% de *accuracy* se tuvo que idear una técnica de boosting basada en fusión del *score*, lo cual demuestra lo útil que son estas técnicas y su buen rendimiento para conseguir un espacio de representación donde lograr una mejor división o clasificación.
 
En el caso de las caras falsas o de los *Deepfake* son un problema bastante complejo si no se conoce los modelos generativos que se han utilizado para generarlas o los modelos árbitros que se han empleado para entrenar las *GANs*, es evidente que lograr incidir en una mejor generación de descriptores ayuda enormemente en la detección de caras falsas.