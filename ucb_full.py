import numpy as np

# standard UCB - functional for full feedback, stochastic setting and weak BB
# no global budget balance for full feedback since we have deterministic
# reward so we do not need to consider actions bringing negative reward, just
# focus on playing profit maximizing action

class UCB:
    def __init__(self, nActions):
        self.nActions = nActions
        self.X = np.zeros(nActions) 
        self.N = 0 # for full feedback, we get the same amount of information 
        # for all arms so N is perceived by the model as equal for all arms 
        self.t = 0
        self.action = None

    def nextAction(self):
        self.t += 1
        if self.t <= self.nActions:
            self.action = self.t-1
        else:
            ucb_values = self.X / self.N   
            self.action = np.argmax(ucb_values)
        return self.action

    def observeReward(self, profits):
        self.X += profits
        self.N += 1
        
        
        
        