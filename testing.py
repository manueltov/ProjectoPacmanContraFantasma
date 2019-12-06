import run
import projIIA_22
from random import *

NUM_TESTES = 5

def corre_pac(param_1, param_2, param_3):

    def pac_22_testing(gState,player):
        #distancia ao fantasma
        a = manhatanDist(gState.board.getPacmanPosition(), gState.board.getGhostPosition(1))
        #fantasma com medo
        # TO DO
        b = 0 #nesta parte tratar caso o fantasma tenha medo ou nao
        #distancia ah pastilha mais proxima
        c = distAhPastilhaMaisProxima(gState,player)
        #distancia ah super pastilha mais proxima
        d = distAhSuperPastilhaMaisProxima(gState,player)
        #ver se vale a pena perseguir o fantasma
        # TO DO
        e = 0
        #quantas jogadas faltam
        f = gState.board.numMoves()/2
        return (a * param_1) + (b * 1) + (c * param_2) + (d * param_3) + (e * 1) + (f * 1)

    return run.runPacmanMain(opt_agent_args = "depth=8,evalFn=testing.corre_pac.pac_22_testing")

def cria_matriz():
    sum = 0
    matriz = [0] * NUM_TESTES
    for i in range(NUM_TESTES):
        matriz[i] = [0,0,0,0]

    for i in range(NUM_TESTES):
        p1 = randint(-100,100)
        p2 = randint(-100,100)
        p3 = randint(-100,100)

        sum = corre_pac(p1, p2, p3)

        matriz[i][0] = sum
        matriz[i][1] = p1
        matriz[i][2] = p2
        matriz[i][3] = p3

    return matriz

def extrai_melhor_params(matriz):
    max_score_index = 0
    for i in range(NUM_TESTES):
        if matriz[i][0] > matriz[max_score_index][0]:
            max_score_index = i
        print("SCORE: " + matriz[max_score_index][0]
              + "P1:"  + matriz[max_score_index][1]
              + "P2:"  + matriz[max_score_index][2]
              + "P3:"  + matriz[max_score_index][3])
    return [matriz[max_score_index][0],
            matriz[max_score_index][1],
            matriz[max_score_index][2],
            matriz[max_score_index][3]] ## return matriz[max_score_index]??

def optimiza():
    matriz = cria_matriz()
    melhores_linha = extrai_melhor_params(matriz)

optimiza()
