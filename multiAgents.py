# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from pacman import Directions


import game



from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

from collections import namedtuple
import random

from utils import argmax

infinity = float('inf')
ClassicPac = namedtuple('ClassicPac', 'to_move, board, extra',defaults=[None])

class ClassicPacman:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor.
    """

    def __init__(self,initial):
        self.initial = initial

    def actions(self, state):
        """Return a list of the allowable moves at this point.
        by default, actions for Pacman, agent index 0.
        ghosts start at 1
        """
       ## print(state.to_move,state.board.numMoves())
        ## print('win:',state.board.isWin())
        if state.to_move == "Pacman":
            return state.board.getLegalActions()
        else:
            return state.board.getLegalActions(1)

    def result(self, state, move,player,extra_fn):
        """Return the state that results from making a move from a state."""
        #print(player,'is thinking and there are still',state.board.numMoves())
        #print('but',state.to_move,'is moving')
        if state.to_move == "Pacman":
            new_board=state.board.generatePacmanSuccessor(move)
            new_extra = state.extra
            if player == 'Ghosts':
                new_extra=extra_fn(new_board,state.extra.copy())
                #print('I have a new extra in my mind',new_extra)
            return ClassicPac(board=new_board,to_move="Ghosts",extra=new_extra)
        else:
            new_board=state.board.generateSuccessor(1,move)
            new_extra=state.extra
            if player == 'Pacman':
                new_extra=extra_fn(new_board,state.extra.copy())
                #print('I have a new extra in my mind',new_extra)
            return ClassicPac(board=new_board,to_move="Pacman",extra=new_extra)



    def utility(self, state, player):
        """Return the value of this final state to player which will be the score."""
        if player == "Pacman":
            return state.board.get_score()
        else:
            return -state.board.get_score()

    def terminal_test(self, state):
        """Return True if this is a final state for the game.
        No actions mean a terminal state.
        When pacman wins or looses there are no legal actions
        """
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print("Next move:",state.to_move)
        print(state.board)


def alphabeta_cutoff_search(state, game, d=4, cutoff_test=None, eval_fn=None,extra_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        #print(depth)
        if cutoff_test(state, depth):
            return eval_fn(state,player)
        v = -infinity
        for a in game.actions(state):              ##
            v = max(v, min_value(game.result(state, a,player,extra_fn),
                                 alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        #print(depth)
        if cutoff_test(state, depth):
            return eval_fn(state,player)
        v = infinity
        for a in game.actions(state):              ##
            v = min(v, max_value(game.result(state, a,player,extra_fn),
                                 alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or
                   (lambda state, depth: depth >= d or
                    game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    extra_fn = extra_fn or (lambda st1: st1.extra)
    #print("Well, I am inside alphabeta and i am going to apply...",extra_fn)
    best_score = -infinity
    beta = infinity
    best_action = None
    movimentos = game.actions(state)  ## jb
    if len(movimentos)==1:
        return movimentos[0]
    else:
        random.shuffle(movimentos)        ## para dar variabilidade aos jogos
        for a in movimentos:              ##
            v = min_value(game.result(state, a,player,extra_fn), best_score, beta, 1)
            if v > best_score:
                best_score = v
                best_action = a
        return best_action



def manhatanDist(p1,p2):
        p1x,p1y=p1
        p2x,p2y=p2
        return abs(p1x-p2x)+abs(p1y-p2y)

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


def identity(gState,extra):
    """ The identity function for the extra part of state
    """
    #print(extra)
    return extra

def extra_mem(gState,extra):
    """ memorizes the positions of Pacman and ghosts
    """
    # print('yeahhhhhh: applying extra_mem')
    if extra == {}:
        n_extra ={'Pacman':[gState.getPacmanPosition()],'Ghosts':[gState.getGhostPosition(1)]}
        return n_extra
    else:
        n_extra=extra.copy()
        n_extra['Pacman']=[gState.getPacmanPosition()]+n_extra['Pacman']
        n_extra['Ghosts']=[gState.getGhostPosition(1)]+n_extra['Ghosts']
    return n_extra

def extra_memF(gState,extra):
    """ memorizes the positions of Pacman and ghosts
    """
    if extra == {}:
        n_extra ={'Pacman':[gState.getPacmanPosition(),gState.getPacmanState().start.getPosition()],'Ghosts':[gState.getGhostPosition(1)]}
        return n_extra
    else:
        n_extra=extra.copy()
        n_extra['Pacman']=[gState.getPacmanPosition()]+n_extra['Pacman']
        n_extra['Ghosts']=[gState.getGhostPosition(1)]+n_extra['Ghosts']
    return n_extra


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







class MultiAgentSearchAgent(Agent):
    """
     This class provides the constructor with additional two attributes: depth and evaluationFunction, and a default name to
    every subclass
    """

   # def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    def __init__(self, index=0,evalFn = 'so_score', extraFn='identity',depth = '2'):
        self.index = index  #0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.extra_fn = util.lookup(extraFn,globals())
        self.depth = int(depth)
        self.extra = {}


    def myName(self):
        # This will be used for the identification of this agent in the record file
        #return self.__class__.__name__ + self.evaluationFunction.__name__
        return self.evaluationFunction.__name__



class AlphaBetaGhost(MultiAgentSearchAgent):
    """
    Your minimax ghost with alpha-beta pruning
    """


    def getAction(self, gameState):
        """
        Returns the alfabeta action using self.depth and self.evaluationFunction
        """
        #print('Before jogada... the extra of Ghost is:',self.extra)
        #print('Jogada dupla:',gameState.numMoves())
        #print("Lets decide with Extra do Ghost",self.extra_fn)
        self.extra=self.extra_fn(gameState,self.extra)
        #print('Current extra:',self.extra)
        currentState = ClassicPac(to_move="Ghosts",board=gameState,extra=self.extra)
        thisGame = ClassicPacman(currentState)
        #print("UAU. here goes alpha beta")
        return alphabeta_cutoff_search(currentState, thisGame, d=self.depth, eval_fn=self.evaluationFunction,extra_fn=self.extra_fn)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
     The alpha-beta pruning Pacman (question 3)
    """


    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        #print('Before jogada... the extra of Pac is:',self.extra)
        #print('Jogada dupla:',gameState.numMoves())
        #print("Lets decide with Extra do Pac",self.extra_fn)
        self.extra=self.extra_fn(gameState,self.extra)
        #print('Current extra:',self.extra)
        currentState = ClassicPac(to_move="Pacman",board=gameState,extra=self.extra)
        thisGame = ClassicPacman(currentState)
        #print("UAU. here goes alpha beta")
        return alphabeta_cutoff_search(currentState, thisGame, d=self.depth, eval_fn=self.evaluationFunction, extra_fn=self.extra_fn)





# ----------------  Random Agents

class RandomPac(Agent):
    """ A pacman that chooses its actions in a random way"""

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        return random.choice(legal)


class RandomGhost(Agent):
    """ A ghost that chooses its actions in a random way"""

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalActions(self.index)
        return random.choice(legal)
