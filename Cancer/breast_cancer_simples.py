#Importação do pandas para iniciar o machine learn
import pandas as pd

#Atributos previsores, base que recebe os valores para serem comparados
previsores = pd.read_csv('entradas-breast.csv')
#Base de comparação, com as respostas a serem observadas e expostas(0 - Benigno/ 1 - Maligno)
#Classificação 1 ou 0 (Binária)
classe = pd.read_csv('saidas-breast.csv')
#426 registros para aprendizagem e 143 para teste

#Importação para a base de erro, ou seja serão 25% para testar e os outros 75% para que a rede consiga treinar e identificar os possiveis padrões
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size=0.25)

#É chamado de sequencial pois é uma sequencia entre camada oculta e camada de saída
import keras
from keras.models import Sequential
#Usada para câmadas densas na rede neural(cada neurônio é ligado com o outro a câmada subsquente(câmada oculta))
from keras.layers import Dense

classificador = Sequential()
#units vai instanciar as camadas ocultas pela formula que defina a quantidade de neuronios
#30 atributos previsores que são 30 entradas ( formula = 30 + (o numero de neuronios na camada de saida)dividido por 2)
# (30 + 1) / 2 = 15.5 arredonda para 16
#Função de ativação relu pois da melhors resultados
#inicializador aleatório, e o imput dim que adiciona quantos elementos tem na camada de entrada (30 atributos previsores)
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30 ))

classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30 ))


#Signifa que como a 2 entradas, se ele retornar perto de 0 é maligno e perto de 1 é benigno
#esta função vai retornar 0 ou 1 como probabilidade
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#Rede Neural criada, uma rede com 30 entradas e aplicação da função relu para a camada oculta e a sigmoide para a de entrada
#Compilar a nossa rede neural
#adam é o parametro da descida do gradiente estocastica (inicia quase sempre com ele)
#loss é a função de perda
#O mais recomendado para ultilizar (entropia binária - para problemas de regressão)
#ultiliza essa função quando esta trabalhando com classificação binaria, o cathegory é para mais de 1 classe
#E mais recomendado por que a formula deste leva em conta o logaritmo, (regressão logistica)
#metrics é um vetor para receber as metricas usadas
#binary_accuracy (pega os registroa certos e os errados)
#epoch são as épocas para os ajustes dos pesos
#batch_siza (pega o conjunto de 10 serviçoes para atualizar)
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100)

#São os 143 registros com as caracteristicas do tumor
previsoes = classificador.predict(previsores_teste)
#se maior que 0.5 é vdd se não falso
previsoes = (previsoes > 0.5)

#compara os dados e retorna como uma matriz 4 por 4, sendo o superior acerto e o superior erro
from sklearn.metrics import confusion_matrix, accuracy_score
#trás a precisao
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#variavel criada para trazer o resultado final dos dados, os valores de teste e os valores de previsão
resultado = classificador.evaluate(previsores_teste, classe_teste)


