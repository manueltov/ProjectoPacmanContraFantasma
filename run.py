import random
import pacman

agent_args = "depth=8,evalFn=projIIA_22.pac_22"
ghost_args = "depth=8,evalFn=anti_score"

listaNomesMapas = ["capsuleClassic", "contestClassic", "mediumClassic", "minimaxClassic", "openClassic", "originalClassic", "powerClassic", "smallClassic", "testClassic", "trappedClassic", "trickyClassic"]

mapa = random.choice(listaNomesMapas)

def runPacmanMain(opt_ghost_args = ghost_args, opt_agent_args = agent_args, opt_mapa = mapa ):
    print("-------------------------------------------------------------")
    print("-- Jogando pacman 'aval_fixe_pac' contra ghost anti_score  --")
    print("-------------------------------------------------------------")
    i = 0
    for i in range(len(listaNomesMapas)):
        opt_mapa = listaNomesMapas[i]
        print("########################################")
        print("## Jogando no mapa: ", opt_mapa, "   ###")
        print("########################################")
        args = pacman.readCommand(["-n", "10", "-q" , "-g", "AlphaBetaGhost", "-b", opt_ghost_args, "-p", "AlphaBetaAgent", "-a", opt_agent_args, "--frameTime", "0", "-k", "1", "-l", opt_mapa] )
        pacman.runGames(**args)


runPacmanMain()
