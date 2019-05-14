#importações necessarias como o pandas para a passagem da base csv
import pandas as pd

#É chamado de sequencial pois é uma sequencia entre camada oculta e camada de saída
from keras.models import Sequential
#Usada para câmadas densas na rede neural(cada neurônio é ligado com o outro a câmada subsquente(câmada oculta))
from keras.layers import Dense
from keras.utils import np_utils

#importação das bases de dados das flores (Iris)
base = pd.read_csv('iris.csv')

#Função que recebe como parâmetro as linhas, para efetuar a divisão dos dados, para armazenar os visores, ja quer a base de dados esta toda unida em uma só 
#vai de 0 a 4, mas na verdade, ele vai de 0 ate 3, ou seja 0, 1, 2, 3
previsores = base.iloc[:, 0:4].values

#criação da classe com os dados, inicialmente como string, sendo visualizada apenas no console e não na atribuição das variáveis
classe = base.iloc[:, 4].values

#Trás os valores modificados em forma de lista e insere-os dentro da variável
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
#especificação das bases de dados
# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1

#Criação da base/classe de treinamento e classe de testes.
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

#Criação dos classificadores de execução e das funções para o treinamento
#Função de ativação relu pois da melhors resultados
classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax'))
#adam é o parametro da descida do gradiente estocastica (inicia quase sempre com ele
#loss é a função de perda
#O mais recomendado para ultilizar (entropia binária - para problemas de regressão)
#binary_accuracy (pega os registroa certos e os errados)
#epoch são as épocas para os ajustes dos pesos
#batch_siza (pega o conjunto de 10 serviçoes para atualizar)
#metrics é um vetor para receber as metricas usadas
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10,
                  epochs = 1000)

#variavel criada para trazer o resultado final dos dados, os valores de teste e os valores de previsão
resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
#trás o valor final, true ou false
previsoes = (previsoes > 0.5)

#usa o maior valor do array, retornando tal mesmo, dentro da variável t do for
#da problema para usar a função matriz por que tem mais de 1 variável, por isso a classeteste2 pega o valor maximo e assim permite criar a matriz
import numpy as np
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

#compara os dados e retorna como uma matriz 4 por 4, sendo o superior acerto e o superior erro
from sklearn.metrics import confusion_matrix
#trás a precisao
matriz = confusion_matrix(previsoes2, classe_teste2)