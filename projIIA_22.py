import searchPlus
import multiAgents
import layout

"esta eh a funcao que fara o pacman pensar em como maximizar o score para si"
def pac_22(gState,player):



    #distancia ao fantasma
    #a = manhatanDist(gState.board.getPacmanPosition(), gState.board.getGhostPosition(1))
    a = distanciaMelhorada(gState, gState.board.getPacmanPosition(), gState.board.getGhostPosition(1))

    #fantasma com medo
    b = 0 #nesta parte tratar caso o fantasma tenha medo ou nao

    #distancia ah pastilha mais proxima
    c = distAhPastilhaMaisProxima(gState,player)

    #distancia ah super pastilha mais proxima
    d = distAhSuperPastilhaMaisProxima(gState,player)

    #return (a * 1) + (b * 1) + (c * 1) + (d * 1)
    #return multiAgents.aval_fixe_pac(gState, player)
    return a

"esta eh a funcao que fara o fantama pensar em como maximizar o score para si"
def fant_22(gState,player):
    return -gState.board.getScore()

"esta funcao que o prof pede eh um extra relativamente ao pacman"
def extraP_22() :
    return 0

"esta funcao que o prof pede eh um extra relativamente ao fantasma"
def extraF_22() :
    return 0



####################################
######   UTILIDADES NOSSAS  ########
####################################

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

"""calcular a distancia tendo em conta os obstaculos"""
def distanciaMelhorada(gState, px, py):

    mapa = layout.Layout(layoutText)
    mapa = limpaLayout(mapa)
    mapa.agentPositions = []

    initialState = mapa.deepCopy()
    initialState.agentPositions.append((0, px))

    finalState = mapa.deepCopy()
    finalState.agentPositions.append((0, py))

    problema = searchPlus.Problem(initialState, finalState)
    distancia = searchPlus.uniform_cost_search(problema).solution()

    return distancia

def limpaLayout(mapa):
    mapa.walls = Grid(self.width, self.height, False)
    mapa.food = Grid(self.width, self.height, False)
    mapa.capsules = []
    mapa.agentPositions = []
    mapa.numGhosts = 0
    mapa.totalFood = len(self.food.asList())
    return mapa


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



############
## here ####
############


def manhatanDist(p1,p2):
        p1x,p1y=p1
        p2x,p2y=p2
        return abs(p1x-p2x)+abs(p1y-p2y)


def minDistPast(gState):
    foodList = gState.board.getFood().asList()
    minDistance = 0
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = gState.board.getPacmanPosition()

        minDistance = min([manhatanDist(myPos, food) for food in foodList])
    return minDistance


def pastilhasARaioX(classicPac, raio):
    foodList = classicPac.board.getFood().asList()
    count = 0
    for food in foodList:
        if(manhatanDist(classicPac.board.getPacmanPosition(), food) < raio):
            count += 1
    return count


def nextGhostPos(classicPac):
    x,y=classicPac.board.getGhostPosition(1)
    dir = classicPac.board.getGhostState(1).getDirection()
    if dir == 'West':
        dx,dy= (-1,0)
    elif dir == 'East':
        dx,dy= (1,0)
    elif dir == 'North':
        dx,dy= (0,1)
    elif dir == 'South':
        dx,dy= (-1,0)
    elif dir == 'Stop':
        dx,dy= (0,0)

    return (x+dx, y+dy)


def ghostTaAfastar(classicPac):
    nextGPostn = nextGhostPos(classicPac)
    nextDist =  manhatanDist(classicPac.board.getPacmanPosition(), nextGPostn)
    return manhatanDist(classicPac.board.getPacmanPosition(), classicPac.board.getGhostPosition(1)) < nextDist

def aval_fixe_pac(classicPac, player):

    myPos = classicPac.board.getPacmanPosition()
    ghostDist = manhatanDist(myPos, classicPac.board.getGhostPosition(1))

    ghostFear = -ghostDist
    minDistPastilha = minDistPast(classicPac)
    score = classicPac.board.getScore()
    QtPast = pastilhasARaioX(classicPac, 20)

    #se ainda o apanho...
    if ghostDist < classicPac.board.getGhostState(1).scaredTimer:
        ghostFear = -1000000000 + (1000/100*ghostDist); ##medo negativo grande, coragem
        print("Pop")

        print("bla: ",classicPac.board.getGhostState(1).start)
        print("ble: ", classicPac.board.getGhostState(1).start.getPosition())
        cx,cy = classicPac.board.getGhostState(1).start.getPosition()
        x = int(cx)
        y = int(cy)
        print(cx , "=", x, " ; ", cy, "=", y)
        return score *1000000000/ghostDist + classicPac.board.numMoves()*100 + len(classicPac.board.getCapsules())*100000000000000000;
    elif ghostDist > 4 and ghostTaAfastar(classicPac) and classicPac.board.getGhostState(1).scaredTimer == 0:
        #menos medo de proximidade de ghost se ta a afastar pq ele n volta pa tras a n ser em cruzilhada
        #dai o > 4
        ghostFear = ghostFear / 2
        #return scooore(classicPac, player)  * 1.5




    ##return 100 * score +  (-0 * ghostFear) + (-1 * minDistPastilha) + (1 * QtPast)
    return scooore(classicPac, player)

def manh_ind(myPos, food):
    return (manhatanDist(myPos, food), food)

def minPast(lista):
    cur_d = 10000000000
    cur_f = (0,0)
    for d,f in lista:
        if d < cur_d :
            cur_f = f
    return cur_f

def aval_fixe_ghost(classicPac, player):

    myPos = classicPac.board.getGhostPosition(1)
    ghostDist = manhatanDist(classicPac.board.getPacmanPosition(), classicPac.board.getGhostPosition(1))
    score = classicPac.board.getScore()

    foodList = classicPac.board.getFood().asList()
    capList = classicPac.board.getCapsules()
    minDistance = 0
    minDistPast = myPos
    capDist = 0
    # Compute distance to the nearest food
    '''
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = classicPac.board.getPacmanPosition()

        minDistPast = minPast([manh_ind(myPos, food) for food in foodList])

    '''
    if len(capList) > 0:
        for cap in capList:
            capDist += -manhatanDist(myPos, cap)

        return -capDist * 0.1
    #se ainda o apanho...
    if 0 < classicPac.board.getGhostState(1).scaredTimer:

        print("Pop")
        print("bla:", classicPac.board.getGhostState(1).start)

        return -(score *1000000000/ghostDist + classicPac.board.numMoves()*100 - len(classicPac.board.getCapsules())*10000) - \
                10000000000000 * manhatanDist(myPos, classicPac.board.getGhostState(1).start.getPosition())

    '''
    if len(foodList) < 100:

        return -score - manhatanDist(myPos, minDistPast)*100000000

    '''

    ##return 100 * score +  (-0 * ghostFear) + (-1 * minDistPastilha) + (1 * QtPast)
    return -scooore(classicPac, player)

def scaredGhost(classicPac, player):

    capList = classicPac.board.getCapsules()
    capDist = 0

    if 0 < classicPac.board.getGhostState(1).scaredTimer:
        return manhatanDist(classicPac.board.getPacmanPosition(), classicPac.board.getGhostPosition(1) )

    if len(capList) == 1:
        for cap in capList:
            capDist += -manhatanDist(classicPac.board.getGhostPosition(1), cap)

        return -capDist * 0.1

    elif len(capList) > 1:
        return manhatanDist(classicPac.board.getPacmanPosition(), classicPac.board.getGhostPosition(1) )

    return -scooore(classicPac, player)

def betterEvaluationFunction(currentGameState, player):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

  I used the following features in this model:
  - distance to the closest active ghost (active ghosts are non-scared ghosts)
  - current score in the game
  - distance to the closest scared ghost
  - number of capsules left
  - number of foods left
  - distance to the closest food
  My evaluation function computes a linear combination of
  these features (or related features, since in some cases, I take
  the inverse of a feature)
  I kept the current score the same, because I saw no reason to modify it.
  I multiply the distance to the closest food by -1.5. This means that the
  larger the distance pac-man has to the closest food, the more negative the
  score is.
  I take the inverse of the distance to the closest active ghost, and then
  multiply it by -2. This means that the larger the distance to the closest
  active ghost, the les negative the score is, but the closer a ghost is,
  the more negative the score becomes.
  I multiply distance to the closest scared ghost by -2, to motivate pac-man to
  move towards scared ghosts. This coefficient is larger than the coefficient I
  used for food, even though the distance to the closest scared ghost will
  almost always be greater than the distance to the nearest food. I chose to
  use a larger coefficient here because:
   - pac-man gets a large number of points for eating a scared ghost
   - if the distance to the closest scared ghost is greater than the distance
     to the nearest food, it is usually because there are not many foods left
     on the board. This means that it's likely more beneficial for pac-man to go
     towards the scared ghost, because eating the scared ghost will likely net
     pac-man more points than eating the remaining foods.
  I multiply the number of capsules left by a very high negative number - -20 -
  in order to motivate pac-man to eat capsules that he passes. I didn't want
  pac-man to move toward capsules over food or over running away from ghosts,
  but I DID want pac-man to eat them when he passed by them. When pac-man
  passes by a capsule, the successor state where pac-man eats a capsule will
  gain +20 points, which is (usually) significant enough that pac-man eats
  the capsule.
  I also multiply the number of foods left by -4, because pac-man should
  minimize the number of foods that are left on the board.
  some results:
  dan at staircar in ~/classes/ai/multiagent on master
  $ python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -n 10 -q
  Pacman emerges victorious! Score: 1366
  Pacman died! Score: -198
  Pacman emerges victorious! Score: 1367
  Pacman emerges victorious! Score: 1737
  Pacman emerges victorious! Score: 1364
  Pacman emerges victorious! Score: 933
  Pacman emerges victorious! Score: 1743
  Pacman emerges victorious! Score: 1193
  Pacman emerges victorious! Score: 1373
  Pacman emerges victorious! Score: 1348
  Average Score: 1222.6
  Scores:        1366, -198, 1367, 1737, 1364, 933, 1743, 1193, 1373, 1348
  Win Rate:      9/10 (0.90)
    Record:        Win, Loss, Win, Win, Win, Win, Win, Win, Win, Win
  dan at staircar in ~/classes/ai/multiagent on master
  $ python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -n 10
  Pacman emerges victorious! Score: 1139
  Pacman emerges victorious! Score: 1362
  Pacman emerges victorious! Score: 1770
  Pacman emerges victorious! Score: 1361
  Pacman emerges victorious! Score: 1234
  Pacman emerges victorious! Score: 1521
  Pacman emerges victorious! Score: 1755
  Pacman emerges victorious! Score: 1759
  Pacman emerges victorious! Score: 1759
  Pacman died! Score: 101
  Average Score: 1376.1
  Scores:        1139, 1362, 1770, 1361, 1234, 1521, 1755, 1759, 1759, 101
  Win Rate:      9/10 (0.90)evaluation
  Record:        Win, Win, Win, Win, Win, Win, Win, Win, Win, Loss
  ---
  I also experimented with using the actual maze-distance instead of the
  manhattan distance, but that turned out to be mostly useless.

  """
  pos = currentGameState.board.getPacmanPosition()
  currentScore = currentGameState.board.getScore()

  if currentGameState.board.isLose():
    return -float("inf")
  elif currentGameState.board.isWin():
    return float("inf")

  # food distance
  foodlist = currentGameState.board.getFood().asList()
  manhattanDistanceToClosestFood = min(map(lambda x: util.manhattanDistance(pos, x), foodlist))
  distanceToClosestFood = manhattanDistanceToClosestFood

  # number of big dots
  # if we only count the number fo them, he'll only care about
  # them if he has the opportunity to eat one.
  numberOfCapsulesLeft = len(currentGameState.board.getCapsules())

  # number of foods left
  numberOfFoodsLeft = len(foodlist)

  # ghost distance

  # active ghosts are ghosts that aren't scared.
  scaredGhosts, activeGhosts = [], []
  for ghost in currentGameState.board.getGhostStates():
    if not ghost.scaredTimer:
      activeGhosts.append(ghost)
    else:
      scaredGhosts.append(ghost)

  def getManhattanDistances(ghosts):
    return map(lambda g: util.manhattanDistance(pos, g.getPosition()), ghosts)

  distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0

  if activeGhosts:
    distanceToClosestActiveGhost = min(getManhattanDistances(activeGhosts))
  else:
    distanceToClosestActiveGhost = float("inf")
  distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)

  if scaredGhosts:
    distanceToClosestScaredGhost = min(getManhattanDistances(scaredGhosts))
  else:
    distanceToClosestScaredGhost = 0 # I don't want it to count if there aren't any scared ghosts

  outputTable = [["dist to closest food", -1.5*distanceToClosestFood],
                 ["dist to closest active ghost", 2*(1./distanceToClosestActiveGhost)],
                 ["dist to closest scared ghost", 2*distanceToClosestScaredGhost],
                 ["number of capsules left", -3.5*numberOfCapsulesLeft],
                 ["number of total foods left", 2*(1./numberOfFoodsLeft)]]

  score = 100    * currentScore + \
          -1.5 * distanceToClosestFood + \
          -2    * (1./distanceToClosestActiveGhost) + \
          -2    * distanceToClosestScaredGhost + \
          -20 * numberOfCapsulesLeft + \
          -4    * numberOfFoodsLeft
  return score




#####################
# in file  pacman ###
#####################
"""
##########################################################################################
if __name__ == 'pacman':
    def corre_pac_main(opt_ghost_args="depth=8,evalFn=anti_score", opt_agent_args="depth=8,evalFn=so_score"):
        args = readCommand(["-q", "-g", "AlphaBetaGhost", "-b", opt_ghost_args, "-p", "AlphaBetaAgent", "-a", opt_agent_args, "--frameTime", "0", "-k", "1", "-l", "mediumClassic"] )
        #print(args['limMoves'])
        #print(args)
        runGames(**args)

        # import cProfile
        # cProfile.run("runGames( **args )")
        pass
##########################################################################################
"""
