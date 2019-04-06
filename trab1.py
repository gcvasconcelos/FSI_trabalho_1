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

def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

# pega a matriz da imagem, divide em n submatrizes, soma o valor total dass submatrizes e armazena no vetor 
def split_and_sum_array(array, n_of_splits):
  new_array = np.empty(shape=(array.shape[0],n_of_splits**2))
  m = 28 // n_of_splits # precisa ser divisivel por 28 (1, 2, 4, 7, 14, 28)
  i = 0
  for element in array:
    subarrays = split(element, m, m)
    j = 0
    for sector in subarrays:
      new_array[i][j] = sector.sum()
      j += 1
    i += 1
  return new_array

x_train = split_and_sum_array(x_train, 4)
x_test = split_and_sum_array(x_test, 4)

# garante que os dados são float e os normaliza
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# aplica LDA nos dados de treino

lda = LinearDiscriminantAnalysis(n_components = 2)
x_lda_train = lda.fit_transform(x_train, y_train) # supervisionado

y_pred = lda.predict(x_test)

# # matriz de confusão e acuracia
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sn

cm = confusion_matrix(y_test, y_pred)  
df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                    columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm)
plt.show()

print('Accuracy ' + str(accuracy_score(y_test, y_pred)))  