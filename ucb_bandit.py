import numpy as np

class UCB:
    def __init__(self, nActions, c=1):
        self.nActions = nActions # number of arms, actions, price pairs
        self.Q = np.zeros(nActions) # average rewards  (here: profits)
        self.X = np.zeros(nActions) # cumulative rewards (here: profits)
        self.N = np.zeros(nActions) # number of arm pulls
        self.t = 0  
        self.action = None
        self.c = c # scaling factor for the exploration term

    def nextAction(self, expected_profits = None):
        """
        Option to include expected_profits estimated in an alternative way 
        in the action selection. Q and expected_profits should converge in 
        the long run, but expected_profits may become more precise earlier on.
        """
        self.t += 1
        # Initial exploration ensuring that each action is selected at least once
        if self.t <= self.nActions:
            self.action = self.t - 1
        
        else: # selection of a price pair maximizing the UCB value
            ucb_values = self.Q + self.c * np.sqrt((2 * np.log(self.t)) / self.N) 
            if expected_profits is None:
                expected_profits = np.zeros(self.nActions)
            self.action = np.argmax(ucb_values + expected_profits)
        return self.action

    def observeReward(self, profit):
        # Parameters' updates
        self.X[self.action] += profit # update the reward vector
        self.N[self.action] += 1 # update the vector of action counts
        # Update the average reward vector
        self.Q[self.action] = self.X[self.action] / self.N[self.action]

    def reset(self):
        self.Q = np.zeros(self.nActions)
        self.X = np.zeros(self.nActions) 
        self.N = np.zeros(self.nActions)
        self.t = 0
        self.action = None

    def results(self):
        return self.acc_profit
    
    
    