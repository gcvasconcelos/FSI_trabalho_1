import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

# usa a lib tensorflow para baixar o dataset MNIST 
# X contem a matriz (28x28) de escala de cinza da imagem (0 a 255)
# Y contem a informação de qual é o número (0 a 9)
# train contem 10.000 imagens e test contem 60.000
(pp_x_train, y_train), (pp_x_test, y_test) = tf.keras.datasets.mnist.load_data()

# função que classifica imagens usando lda, mede a acurácia e printa matriz de confusão
def lda_classifier(x_train, y_train, x_test, y_test, dimensions):
  # aplica LDA nos dados de treino
  lda = LinearDiscriminantAnalysis(n_components = dimensions)
  x_lda_train = lda.fit_transform(x_train, y_train) # supervisionado

  y_pred = lda.predict(x_test)

  print('Accuracy LDA Algorithm: ' + str(accuracy_score(y_test, y_pred)))

  cm = confusion_matrix(y_test, y_pred)  
  df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                      columns = [i for i in "0123456789"])
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm)
  plt.show()

# função que classifica imagens usando knn, mede a acurácia e printa matriz de confusão
def knn_classifier(x_train, y_train, x_test, y_test, n):
  knn = KNeighborsClassifier(n_neighbors=n)
  x_knn_train = knn.fit(x_train, y_train)

  y_knn_pred = knn.predict(x_test)

  print('Accuracy K-nn Algorithm with 2 neighbor(s): ' + str(accuracy_score(y_test, y_knn_pred)))

  cm = confusion_matrix(y_test, y_knn_pred)
  df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                      columns = [i for i in "0123456789"])
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm)
  plt.show()

  knn = KNeighborsClassifier(n_neighbors=15)
  x_knn_train = knn.fit(x_train, y_train)

  y_knn_pred = knn.predict(x_test)

  print('Accuracy K-nn Algorithm with 15 neighbor(s): ' + str(accuracy_score(y_test, y_knn_pred)))

  cm = confusion_matrix(y_test, y_knn_pred)
  df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                      columns = [i for i in "0123456789"])
  plt.figure(figsize = (10,7))
  sn.heatmap(df_cm)
  plt.show()

# divide matriz em vetor de submatrizes
def split(array, nrows, ncols):
    return(array.reshape(array.shape[1]//nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

# pega a matriz da imagem, divide em n submatrizes, soma o valor total dass submatrizes e armazena no vetor 
def split_and_sum_image(images, n_of_splits):
  new_images = np.empty(shape=(images.shape[0],n_of_splits**2))
  m = 28 // n_of_splits # precisa ser divisivel por 28 (1, 2, 4, 7, 14, 28)
  i = 0
  for image in images:
    subimages = split(image, m, m)
    j = 0
    for sector in subimages:
      new_images[i][j] = sector.sum()
      j += 1
    i += 1
  return new_images

# pega a média do centro da imagem e adiciona a lista de critérios
def get_center_mean(image, array):
  new_array = np.empty(shape=(array.shape[0], array.shape[1]+1))
  i = 0
  for element in image:
    center_mean = int(np.mean(element[10:16,10:16]))
    new_array[i] = np.append(array[i], center_mean)
    i += 1
  return new_array

# agrega toda as funções de processamento 
def preprocessing(pp_x_train, pp_x_test):
  x_train = split_and_sum_image(pp_x_train, 14)
  x_test = split_and_sum_image(pp_x_test, 14)
  x_train = get_center_mean(pp_x_train, x_train)
  x_test = get_center_mean(pp_x_test, x_test)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  sc = StandardScaler()
  x_train = sc.fit_transform(x_train)
  x_test = sc.transform(x_test)
  
  return [x_train, x_test]

x_train, x_test = preprocessing(pp_x_train, pp_x_test)

lda_classifier(x_train, y_train, x_test, y_test, 2)
knn_classifier(x_train, y_train, x_test, y_test,2)
