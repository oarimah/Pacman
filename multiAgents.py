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
#code filled in by Ositadinma Arimah

import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        foodList = newFood.asList()
        ghostPosition = successorGameState.getGhostPositions()
        foodDistance = []
        ghostDistance = []

        for food in foodList:
          foodDistance.append(manhattanDistance(food, newPos))

        for ghost in ghostPosition:
          ghostDistance.append(manhattanDistance(ghost, newPos))

        if currentGameState.getPacmanPosition() == newPos:
          return (-(float("inf")))

        for ghostDistance in ghostDistance:
          if ghostDistance < 2:
            return (-(float("inf")))

        if len(foodDistance) == 0:
          return float("inf")
        else:
          minimumFoodDistance = min(foodDistance)
          maximumFoodDistance = max(foodDistance)

        return 1000/sum(foodDistance) + 10000/len(foodDistance)

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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"

        def maxState(state,depth, agentIndex):
          depth -= 1
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          val = float("-inf")
          for action in state.getLegalActions(agentIndex):
            succNode = state.generateSuccessor(agentIndex,action)
            score = minState(succNode,depth,agentIndex+1)[0]
            if score > val:
              maximumAction = action
              val = score
          return (val,maximumAction)

        def minState(state,depth,agentIndex):
          if depth < 0 or state.isLose() or state.isWin():
            return (self.evaluationFunction(state),None)
          v = float("inf")
          beta, succAgent = (minState, agentIndex+1) if agentIndex < state.getNumAgents()-1 else (maxState, 0)
          for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            score = beta(successor,depth,succAgent)[0]
            if score < v:
              minimumAction = action
              v = score
          return (v,minimumAction)
        return maxState(gameState,self.depth,0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


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

        def value(agentIndex, depthLeft, state):
          if depthLeft == 0 and agentIndex == 0:
              return valueAction(self.evaluationFunction(state), None)
          return maxValue(agentIndex, depthLeft-1, state) if agentIndex == 0 else expectimaxValue(agentIndex,depthLeft, state)

        def maxValue(agentIndex, depthLeft, state):
          valAction = valueAction(float("-inf"), None)
          if len(state.getLegalActions(agentIndex)) == 0:
              return value(0,0,state)
          for action in state.getLegalActions(agentIndex):
            val = value((agentIndex+1)%state.getNumAgents(), depthLeft, state.generateSuccessor(agentIndex,action))
            if val.value > valAction.value: valAction = valueAction(val.value, action)
          return valAction


        def expectimaxValue(agentIndex,depthLeft,state):
          num = len(state.getLegalActions(agentIndex))
          if num == 0:
              return value(0,0,state)
          val = sum([value((agentIndex+1)%state.getNumAgents(), depthLeft,state.generateSuccessor(agentIndex,action)).value
                  for action in state.getLegalActions(agentIndex)]) / float(num)
          return valueAction(val, None)

        return value(0, self.depth, gameState).action



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# value action object
class valueAction:

  def __init__(self, value, action):# initialize the varibales
    self.value = value
    self.action = action

# Abbreviation
better = betterEvaluationFunction
