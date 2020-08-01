# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
Project by James Kang and Henzi Kou
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Find the start position
    start = problem.getStartState()

    # Initalize a fringe using stack properties to store states
    fringe = util.Stack()

    # Add the start position to the visited list
    visited = []
    visited.append(start)

    # Format start state as a tuple to append to fringe
    fringe.push((start, []))

   # Traverse to next state in fringe while fringe is not empty
    while not fringe.isEmpty():
        # Take next location in the fringe
        # Pop the location tuple and action list
        location, action = fringe.pop()

        if problem.isGoalState(location):
            return action

        # Add location to visited list
        visited.append(location)

        # Add location's successors to fringe
        successors = problem.getSuccessors(location)

        for i in successors:
            if i[0] not in visited:
                direction = i[1]
                 # If successors have not been visited before then add to fringe
                 # Add direction to action list
                fringe.push((i[0], action + [direction]))

    return action + [direction]

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # Initialize variables
    start = problem.getStartState()
    current = problem.getStartState()
    visited = []
    fringe = util.Queue()

    # Check if start state is the goal state
    if problem.isGoalState(start) is True:
        return start

    # Append start state to visited list
    visited.append(start)

    fringe.push((start, []))

    # Traverse to next state in fringe while fringe is not empty
    while not fringe.isEmpty():
        # Get location and action list
        location, action = fringe.pop()

        # Check if current location is goal state
        if problem.isGoalState(location):
            return action

        # Retrieve successors
        successors = problem.getSuccessors(location)

        for i in successors:
            if i[0] not in visited:
                current = i[0]
                direction = i[1]

                # Add location of successor i, to visited
                visited.append(i[0])

                # Unvisited successors are added to fringe
                fringe.push((i[0], action + [direction]))

    return action

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Initialize variables
    start = problem.getStartState()
    visited = []
    fringe = util.PriorityQueue()

    # Push start start into fringe including priority cost
    fringe.push((start, []), 0)

    # While the fringe is not empty loop through each state
    while not fringe.isEmpty():
        # Get location and action list
        location, action = fringe.pop()

        # Check if current location is goal state
        if problem.isGoalState(location):
            return action

        if location not in visited:
            successors = problem.getSuccessors(location)
            for s in successors:
                # Loop through each successor
                if s[0] not in visited:
                    direction = s[1]
                    cost = action + [direction]
                    fringe.push((s[0], cost), problem.getCostOfActions(cost))

        visited.append(location)

    return action

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # Initialize variables
    start = problem.getStartState()
    visited = []
    fringe = util.PriorityQueue()

    # Add start state to fringe
    fringe.push((start, []), nullHeuristic(start, problem))

    # Loop through fringe while there are items
    while not fringe.isEmpty():
        # Get location and action list
        location, action = fringe.pop()

        # Check if current location is goal state
        if problem.isGoalState(location):
            return action

        # Current state must not be a visited state
        if location not in visited:
            # Obtain successors
            successors = problem.getSuccessors(location)
            for s in successors:
                if s[0] not in visited:
                    direction = s[1]
                    actions = action + [direction]
                    # Combined heuristic cost
                    hCost = problem.getCostOfActions(actions) + heuristic(s[0], problem)
                    fringe.push((s[0], actions), hCost)

        visited.append(location)

    return action

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
