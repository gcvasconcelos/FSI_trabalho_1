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

x_train = x_train.reshape((x_train.shape[1]*x_train.shape[2]), x_train.shape[0]).transpose()
x_test = x_test.reshape((x_test.shape[1]*x_test.shape[2]), x_test.shape[0]).transpose()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# aplica LDA nos dados de treino
lda = LinearDiscriminantAnalysis(n_components = 2 )
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

# # abre uma imagem e seu label
# print(y_train[0])
# plt.imshow(x_train[0], cmap='Greys')
# plt.show()