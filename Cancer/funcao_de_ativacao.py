import numpy as np
#Transfer Function

#Para problemas linearmente separados
def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

#Usado para problemas binarios (0/1)
def sigmoideFunction(soma):
    return 1 / (1 + np.exp(-soma))

#Para classificação também, pois trás de -1 e 1
def tahnFunction(soma):
    return(np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))
    
#Sempre para cima com um valor positivo      
def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0

#Função Linear, usada para regressão
def linearFunction(soma):
    return soma

#Função da probabilidade, classificação com mais de 1 classe
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

teste = stepFunction(-1)
teste = sigmoideFunction(2.1)
teste = tahnFunction(2.1)
teste = reluFunction(-2)
teste = linearFunction(2.1)
valores = [5.0,2.0,1.3]
print(softmaxFunction(valores))






