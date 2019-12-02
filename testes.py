import pacman
from random import *

def corre_pac(param_1, param_2, param_3):

    def func_aval():
        p1 = param_1
        p2 = param_2
        p3 = param_3
        ##logica de avaliacao com params
        return p1*p2*p3

    ##correr com os atributos q queremos lalala // funcao q o stor usa pa correr

    return pacman.corre_pac_main()

def cria_matriz():
    sum = 0
    ##cria matriz[100][4]
    matriz = [0] * 100
    for i in range(100):
        matriz[i] = [0,0,0,0]

    for i in range(100):
        p1 = randint(0,100)
        p2 = randint(0,100)
        p3 = randint(0,100)

        for j in range(100):
            sum += corre_pac(p1, p2, p3)

        sum = sum / 100
        matriz[i][0] = sum
        matriz[i][1] = p1
        matriz[i][2] = p2
        matriz[i][3] = p3

    return matriz

def extrai_melhor_params(matriz):
    max_score_index = 0
    for i in range(100):
        if matriz[i][0] > matriz[max_score_index][0]:
            max_score_index = i
            print("SCORE: " + matriz[max_score_index][0] + "P1:"  + matriz[max_score_index][1] + "P2:"  + matriz[max_score_index][2] + "P3:"  + matriz[max_score_index][3])
    return [matriz[max_score_index][0], matriz[max_score_index][1], matriz[max_score_index][2], matriz[max_score_index][3]] ## return matriz[max_score_index]??


def optimiza():
    ##melhores_linhas=[0,0,0,0]
    for i in range(100):
        matriz = cria_matriz()
        melhores_linhas[i] = extrai_melhor_params(matriz)

    melhor_linha
    sum = 0
    for i in range(3):
        for j in range(100):
            sum += melhores_linhas[j][i+1]
        melhor_linha[i+1] = sum/100

    melhor_linha[0] =  corre_pac(melhor_linha[1], melhor_linha[2], melhor_linha[3])
    print("SCORE: " + melhor_linha[0] + "P1:"  +melhor_linha[1] + "P2:"  + melhor_linha[2] + "P3:"  +melhor_linha[3])
    print("Outra vez?? (s/n)")
    answer = input()
    while answer == s:
        melhor_linha[0] =  corre_pac(melhor_linha[1], melhor_linha[2], melhor_linha[3])
        print("SCORE: " + melhor_linha[0] + "P1:"  +melhor_linha[1] + "P2:"  + melhor_linha[2] + "P3:"  +melhor_linha[3])
        print("Outra vez?? (s/n)")
        answer = input()




optimiza()
"""
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

$ python test.py arg1 arg2 arg3

Number of arguments: 4 arguments.
Argument List: ['test.py', 'arg1', 'arg2', 'arg3']
"""
