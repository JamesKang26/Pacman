# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
          # Loop through all iterations
          counter = util.Counter()
          for state in self.mdp.getStates():
            # Determine if it's the terminal state, if so then set counter to 0 at given state
            if self.mdp.isTerminal(state):
              counter[state] = 0
            else:
              # Else take the max qValue and set as the counter at given state
              values = []
              actions = self.mdp.getPossibleActions(state)
              
              for action in actions:
                qValue = self.computeQValueFromValues(state, action)
                values.append(qValue)
                counter[state] = max(values)

          self.values = counter   # Update self.values dict


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # Initialize qValue
        qValue = 0

        # Obtain all transition states and probabilities
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        for nextState, prob in statesAndProbs:
          reward = self.mdp.getReward(state, action, nextState) # Calculate reward
          qValue += prob * (reward + self.discount * self.values[nextState]) # Calculate qValue

        return qValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
          # Check if state is terminal state
          return None

        # Obtain all possible actions of current state and determine the
        # best action by passing into computeQValueFromValues function.
        actions = self.mdp.getPossibleActions(state)
        bestAction = max(actions, key=lambda action: self.computeQValueFromValues(state, action))

        return bestAction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        size = len(states)

        for i in range(self.iterations):
          # Loop through all iterations
          state = states[i % size] # Cyclic value iteration for each unique i

          if self.mdp.isTerminal(state):
            self.values[state] = 0
          else:
            values = [] # List of all values
            
            for action in self.mdp.getPossibleActions(state):
              # Compute the qValues of each (state, action) and append to values list
              qValue = self.computeQValueFromValues(state, action)                
              values.append(qValue)
            
            self.values[state] = max(values) # Assign the max value to current state


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Initialize variables 

        states = self.mdp.getStates()
        predecessors = {} # Dictionary of predecessors
        pQueue = util.PriorityQueue() # empty priority queue

        for s in states: # compute predecessors of all states
          predecessors[s] = set()


        for s in states: # for each non- terminal state s
          count1 = util.Counter()
          for act in self.mdp.getPossibleActions(s):
            trans = self.mdp.getTransitionStatesAndProbs(s, act)
            for (nextState, prob) in trans:
              if(prob != 0):
                predecessors[nextState].add(s)
            count1[act] = self.computeQValueFromValues(s, act)

          if not self.mdp.isTerminal(s): 
            Qmax = count1[count1.argMax()]  # finding out maxQ
            diff = abs(self.values[s] - Qmax)  #fidning the difference
            pQueue.update(s, -diff)  # updating out queue (negative because our queue is a min heap)
        
        for i in range(self.iterations):  #for iteration
          if pQueue.isEmpty():  #if the queue is empty then terminate
            return

          s = pQueue.pop()  # popping a state s off of the queue

          if not self.mdp.isTerminal(s):
            count1 = util.Counter()
            for act in self.mdp.getPossibleActions(s):
              count1[act] = self.computeQValueFromValues(s, act)
            self.values[s] = count1[count1.argMax()] #updating the values

          for pre in predecessors[s]:
            count2 = util.Counter()
            for act in self.mdp.getPossibleActions(pre):
              count2[act] = self.computeQValueFromValues(pre, act)

            Qmax2 = count2[count2.argMax()]
            diff = abs(self.values[pre] - Qmax2)

            if diff > self.theta:    #pushing pre into the queue
              pQueue.update(pre, -diff)






