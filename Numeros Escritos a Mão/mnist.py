#Importação necessaria, essa importação não leva base de dados, por que a base é importada pelo próprio keras
#o matplot também vai ser importada para plotar os números
#Possui 10 classes
import matplotlib.pyplot as plt
from keras.datasets import mnist #base com os dados especificos (imagens de numeros)
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout# base que modifica os dados para que estes sejam transformados de matriz para vetor
from keras.utils import np_utils #usado para o mapeamento de variavels do tipo dame, ou seja com mais classes (10 classes)
from keras.layers import Conv2D, MaxPooling2D #convulação baseada em 4 etapas
from keras.layers.normalization import BatchNormalization


(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data() #definição de variaveis para testar e treinar a base de dados, vai carregar a base de dados
plt.imshow(X_treinamento[0], cmap = 'gray')#responsavel por imprimir (visualizar) os registros, tendo o indice e a cor desejava (mais cor atrapalha o processamento)
plt.title('Classe ' + str(y_treinamento[0])) #classe que esta sendo ultilizada, imprimindo o titulo


previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],#modificação para que o tensorflow/keras possa identificar os dados
   
   28, 28, 1)#parametros de entrada assim como parametro de 1 canal de cor
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)#estrada com os mesmos dados para o teste da rede e não o treinam,ento como a outra
previsores_treinamento = previsores_treinamento.astype('float32')#aumenta a variavel de uint8 para float 32. Variavel de treinamento e de teste logo abaixo
previsores_teste = previsores_teste.astype('float32')


previsores_treinamento /= 255 #minimo 0 maximo 255 (diminue os dados das cores ja que estavam muito grandes)
previsores_teste /= 255 #o mesmo para o teste


classe_treinamento = np_utils.to_categorical(y_treinamento, 10)#cria os 10 parametros e transforma os dados de antes de encoding do tipoo 000, 100,200 e etc
classe_teste = np_utils.to_categorical(y_teste, 10)#o mesmo para o teste


classificador = Sequential()
classificador.add(Conv2D(32, (3,3), #Mapa de caracteristicas, tendo como parametro a quantidade de filtoros e detector de caracteristicas (32 paras de caracteres)
                         input_shape=(28, 28, 1),#tamanho do detector de cxaracteristicas (1 pixel pra direita e para baixo
                         activation = 'relu'))#função relu só de subida
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))#pega uma parte do mapa(matriz) e separa o maior valor
#classificador.add(Flatten())#modifica para dense
#tamanho do detector de cxaracteristicas (1 pixel pra direita e para baixo
#função relu só de subida
classificador.add(Conv2D(32, (3,3), activation = 'relu'))#Mapa de caracteristicas, tendo como parametro a quantidade de filtoros e detector de caracteristicas (32 paras de caracteres)

classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))


classificador.add(Flatten())#classificador.add(Flatten())#modifica para dense

#camada escondida da rede neural densa com as funções de ativação relu e softmax
classificador.add(Dense(units = 128, activation = 'relu'))#função relu só de subida
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))#função relu só de subida
classificador.add(Dropout(0.2))
#adam é o parametro da descida do gradiente estocastica (inicia quase sempre com ele
#loss é a função de perda
#O mais recomendado para ultilizar (entropia binária - para problemas de regressão)
#binary_accuracy (pega os registroa certos e os errados)
#epoch são as épocas para os ajustes dos pesos
#batch_siza (pega o conjunto de 10 serviçoes para atualizar)
#metrics é um vetor para receber as metricas usadas
classificador.add(Dense(units = 10, 
                        activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 2,#(quantidade de epocas de execução)
                  validation_data = (previsores_teste, classe_teste))


resultado = classificador.evaluate(previsores_teste, classe_teste)#variável de previsão de testes e classificação de testes
