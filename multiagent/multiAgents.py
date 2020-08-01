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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return 999999999999999999999999

        g_position = currentGameState.getGhostPosition(1)  # ghost position 
        d_f_ghost = util.manhattanDistance(g_position, newPos)  # distance from the ghost

        value = max(d_f_ghost, 7) + successorGameState.getScore()
        foods = newFood.asList()
        c_food = 300 # some random large number

        for food in foods:
            dist = util.manhattanDistance(food, newPos)
            if(dist < c_food):
                c_food = dist

        if(currentGameState.getNumFood() > successorGameState.getNumFood()):
            value += 300

        if action == Directions.STOP:
            value -= 7

        value -= 7 * c_food

        caps = currentGameState.getCapsules()
        if successorGameState.getPacmanPosition() in caps:
            value += 150


        return value

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agentIndex):
            """
            Implementing minimax.
            """
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                # Check if depth is reached or game is won/loss
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                # Maximize for pacman
                maximum = -float("inf") # Return value
                actions = gameState.getLegalActions(agentIndex) # Get all possible actions

                for action in actions:
                    # Loop through all the possible states and return the highest value state
                    currState = gameState.generateSuccessor(agentIndex, action)

                    if minimax(currState, depth, 1) > maximum:
                        # Update the new max state
                        maximum = minimax(currState, depth, 1)

                return maximum

            else:
                # Minimize for ghosts
                minimum = float("inf")  # Return value
                actions = gameState.getLegalActions(agentIndex) # Get all possible actions
                nextAgent = agentIndex + 1  # Increase the agent index

                # Check the next agent index and update the index and depth accordingly
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1

                for action in actions:
                    # Loop through all the possible states and return the lowest value state
                    currState = gameState.generateSuccessor(agentIndex, action)

                    if minimax(currState, depth, nextAgent) < minimum:
                        # Update the new min state
                        minimum = minimax(currState, depth, nextAgent)

                return minimum

        bestAction = -float("inf")
        action = Directions.WEST

        for state in gameState.getLegalActions(0):
            # Maximize the action for pacman at the root node
            actionValue = minimax(gameState.generateSuccessor(0, state), 0, 1)
            if actionValue > bestAction:
                bestAction = actionValue
                action = state

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        Infinity = float("inf")

        def maxValue(state, depth, a, b):
            v = -Infinity   # Return value v
            actions = state.getLegalActions(0)  # Get all possible actions

            if not actions or depth == self.depth:
                return self.evaluationFunction(state)

            if depth == 0:
                # Check depth for pruning
                bestAction = actions[0]

            for action in actions:
                # Check each action and prune based off of alpha and beta values
                newState = state.generateSuccessor(0, action)
                newV = minValue(newState, 0 + 1, depth + 1, a, b) # Obtain new min value

                if newV > v:
                    v = newV
                    if depth == 0:
                        bestAction = action
                if v > b:
                    return v
              
                a = max(a, v)

            if depth == 0:
                return bestAction

            return v

        def minValue(state, agentIndex, depth, a, b):
            v = Infinity    # Return value v
            actions = state.getLegalActions(agentIndex) # Get all possible actions

            if not actions:
                return self.evaluationFunction(state)

            for action in actions:
                # Check each action and prune based off of alpha and beta values
                newState = state.generateSuccessor(agentIndex, action)

                if agentIndex == state.getNumAgents() - 1:
                    # Check if it's the last ghost. If so then find new max value.
                    newV = maxValue(newState, depth, a, b)
                else:
                    # If not then find new min value
                    newV = minValue(newState, agentIndex + 1, depth, a, b)

                v = min(v, newV)

                if v < a:
                    return v

                b = min(b, v)

            return v
        
        bestAction = maxValue(gameState, 0, -Infinity, Infinity)
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxi(gameState, depth):  # the maximinzing function
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            value = -999999999999999999999999
            legal_act = gameState.getLegalActions(0)

            for act in legal_act:          
                child = gameState.generateSuccessor(0,act)
                comp = expect(child, depth, 1)
                if(value < comp):
                    value = comp
            return value


        def expect(gameState, depth, ind):  # the expected function 
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)


            legal_act = gameState.getLegalActions(ind)
            value = 0
            num_actions = len(legal_act)
            num_ghosts = gameState.getNumAgents() - 1 

            for act in legal_act:
                child = gameState.generateSuccessor(ind, act)
                if(ind == num_ghosts):
                    value += maxi(child, depth -1)
                else:
                    value += expect(child, depth, ind +1)

            return value / num_actions

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        legal_act = gameState.getLegalActions(0)
        direction = Directions.STOP
        value = -999999999999999999999999

        for state in legal_act:
            child = gameState.generateSuccessor(0, state)
            temp = expect(child, self.depth, 1)
            if temp > value:
                value = temp
                direction = state

        return direction






        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    Basically we put in alot of different things in consideration the prioity of these constraints are as follows

    1) the main thing we would like to avoid is getting killed. With this as our number one constraint, if our little 
    man see's a ghost within 4 steps (legal steps) away from him he will move away from the said ghost shown as the d_ghost

    2) the next thing in our agenda is to make sure that the game eventualy ends. and we know it ends when the pacman 
    eats all of the food dots. With this us our next biggest concern we take the nearest food item and to make sure that the father away the
    food is the less points we get we subtract the value with the closest food * 1.5

    3) The next thing we have to worry about is the fact that the pacman seems to get stuck sometimes, especally when the closet food is across the game
    board the pacman seems to not be able to go towards it, it's probably because the penalty for the food being far away adds up with a greater distance
    to fix this i decided that I would give a penaty to the pacman by the number of food remaining * 3 so that we would get a better score if he ate all
    of the food dots

    4) The next priority was the power pills that pacman gets to have. This is quite useful because when the pacman goes into his power mode he is
    able to ignore the ghost (and possibly earn extra points for killing them???) and focus on getting the food. Although it's not quite neccaasry it
    helps to improve the score of the pac man

    With the provided constraints I was able to get the average score of 1096.4
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 999999999999999999999999
    if currentGameState.isLose():
        return -999999999999999999999999

    value = scoreEvaluationFunction(currentGameState)

    # 1
    d_ghost = 999999999999999999999999
    n_ghost = currentGameState.getNumAgents() - 1
    val = 1

    while(val <= n_ghost):
        n_dist = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(val))
        if(n_dist < d_ghost):
            d_ghost = n_dist
        val += 1

    value += max(d_ghost, 4)

    # 2
    newFood = currentGameState.getFood()
    food_list = newFood.asList()
    closest_f = 999999999999999999999999

    for food in food_list:
        dist = util.manhattanDistance(food, currentGameState.getPacmanPosition())
        if(dist < closest_f):
            closest_f = dist
    value -= closest_f * 1.5

    #3
    value -= 3 * len(food_list)

    #4
    cap_loc = currentGameState.getCapsules()
    value -= 3.5 * len(cap_loc)


    return value

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction























