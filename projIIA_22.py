"esta eh a funcao que fara o pacman pensar em como maximizar o score para si"
def pac_22(gState,player):

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

    return (a * 1) + (b * 1) + (c * 1) + (d * 1) + (e * 1) + (f * 1)

"esta eh a funcao que fara o fantama pensar em como maximizar o score para si"
def fant_22(gState,player):
    return -gState.board.getScore()

"esta funcao que o prof pede eh um extra relativamente ao pacman"
def extraP_22() :
    return 0

"esta funcao que o prof pede eh um extra relativamente ao fantasma"
def extraF_22() :
    return 0



##############################
######   UTILIDADES   ########
##############################

"""
-> calcula a distancia ah pastilha mais proxima
-> atualmente usa o manhattanDistance
-> ista funcao eh basicamente uma copia da funcao scooore() dado pelo prof
"""
def distAhPastilhaMaisProxima(gState,player):
    foodList = gState.board.getFood().asList()
    minDistance = 0
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = gState.board.getPacmanPosition()
        minDistance = min([manhatanDist(myPos, food) for food in foodList])
    return minDistance

"""
-> calcula a distancia ah super pastilha mais proxima
-> atualmente usa o manhattanDistance
-> ista funcao eh quase uma copia da funcao distAhPastilhaMaisProxima()
"""
def distAhSuperPastilhaMaisProxima(gState,player):
    superFoodList = gState.board.getCapsules()
    minDistance = 0
    # Compute distance to the nearest food
    if len(superFoodList) > 0: # This should always be True,  but better safe than sorry
        myPos = gState.board.getPacmanPosition()
        minDistance = min([manhatanDist(myPos, superFood) for superFood in superFoodList])
    return minDistance

"""
num futuro podemos ter aqui uma melhor que o manhattanDistance
uma função que pode funcionar tipo algoritmo A* a percorrer o board todo a encontrar
o melhor caminho com base na heuristica manhattanDistance
"""
def manhatanDist(p1,p2):
    p1x,p1y=p1
    p2x,p2y=p2
    return abs(p1x-p2x)+abs(p1y-p2y)



####################################################
######   DO PROF MAS PARA IR TENDO IDEIAS   ########
####################################################
"""
def so_score(gState,player):
    return gState.board.getScore()

def anti_score(gState,player):
    return -gState.board.getScore()

def scooore(gState,player):
    foodList = gState.board.getFood().asList()
    minDistance = 0
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = gState.board.getPacmanPosition()
        minDistance = min([manhatanDist(myPos, food) for food in foodList])
    return gState.board.getScore() * 100 - minDistance

def manhatanDist(p1,p2):
        p1x,p1y=p1
        p2x,p2y=p2
        return abs(p1x-p2x)+abs(p1y-p2y)
"""
