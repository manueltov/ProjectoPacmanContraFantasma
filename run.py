import pacman

agent_args = "depth=8,evalFn=projIIA_22.pac_22"
ghost_args = "depth=8,evalFn=anti_score"

def runPacmanMain(opt_ghost_args = ghost_args, opt_agent_args = agent_args):
    args = pacman.readCommand(["-n", "2", "-q", "-g", "AlphaBetaGhost", "-b", opt_ghost_args, "-p", "AlphaBetaAgent", "-a", opt_agent_args, "--frameTime", "0", "-k", "1", "-l", "mediumClassic"] )
    pacman.runGames(**args)

runPacmanMain()
