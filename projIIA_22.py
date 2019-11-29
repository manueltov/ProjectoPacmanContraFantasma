def pac_22_matov(gState,player):
    foodList = gState.board.getFood().asList()
    minDistance = 0
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
        myPos = gState.board.getPacmanPosition()

        minDistance = min([manhatanDist(myPos, food) for food in foodList])
    return gState.board.getScore() * 100 - minDistance

def fant_22_matov(gState,player):
    return -gState.board.getScore()

def extraP_22_matov :
    return

def extraF_22_matov :
    return
