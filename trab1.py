import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# usa a lib tensorflow para baixar o dataset MNIST 
# X contem a matriz (28x28) de escala de cinza da imagem (0 a 255)
# Y contem a informação de qual é o número (0 a 9)
# train contem 10.000 imagens e test contem 60.000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# tranforma a imagem 28x28 em um vetor de 784
def threeD_to_twoD(array):
  return array.reshape((array.shape[1]*array.shape[2]), array.shape[0]).transpose()

x_train = threeD_to_twoD(x_train)
x_test = threeD_to_twoD(x_test)

# pega regioes de cada imagem e soma seus valores em escala de cinza
def split_and_sum_array(array, n_of_splits):
  new_array = np.empty(shape=(array.shape[0],n_of_splits))
  i = 0
  for big_array in array:
    subarrays = np.split(big_array, n_of_splits)
    j = 0
    for subarray in subarrays:
      new_array[i][j] = subarray.sum()
      j += 1
    i += 1
  return new_array

x_train = split_and_sum_array(x_train, 16)
x_test = split_and_sum_array(x_test, 16)

# garante que os dados são float e normaliza

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# # aplica LDA nos dados de treino
# lda = LinearDiscriminantAnalysis(n_components = 2 )
# x_lda = lda.fit_transform(x_train, y_train)
# print(x_lda)

# x_test = lda.transform(x_test)

# markers = ['s','x','o']
# colors = ['r','g','b']
# fig = plt.figure(figsize=(10,10))
# ax0 = fig.add_subplot(111)
# for l,m,c in zip(np.unique(y_train),markers,colors):
#   ax0.scatter(x_train[:,0][y_train==l],x_train[:,1][y_train==l],c=c,marker=m)

# # matriz de confusão e acuracia
# from sklearn.metrics import confusion_matrix  
# from sklearn.metrics import accuracy_score

# cm = confusion_matrix(y_test, y_pred)  
# print(cm)  
# print('Accuracy ' + str(accuracy_score(y_test, y_pred)))  

# # abre uma imagem e seu label
# print(y_train[0])
# plt.imshow(x_train[0], cmap='Greys')
# plt.show()