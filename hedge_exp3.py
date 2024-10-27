import numpy as np

np.random.seed(123)

# Source of inspiration: andCelli github


class Hedge:
    def __init__(self, learning_rate, nActions):
        if learning_rate is None: # if unspecified, default value assigned
            learning_rate = np.sqrt(2 * np.log(nActions) / nActions) # or without 2
        self.learning_rate = learning_rate # learning rate in (0, 1]
        self.nActions = nActions
        self.w = np.full(nActions, 1. / nActions)[:,np.newaxis] # weights
        self.p = np.full(nActions, 1. / nActions)[:,np.newaxis] # probabilities
        self.action = None

    def nextAction(self, indices = None): # indices array gives an option to specify 
        # the subset of arms to choose from (e.g. respecting global budget balance)
        if indices is None: # by default, all actions can be selected
            indices = np.arange(self.nActions)
            self.p = self.w
        else: # probability vector adjusted to the subset of arms: vector of
        # conditional probabilities given belonging to indices
            self.p = self.w[indices] / np.sum(self.w[indices]) 
        # Probabilistic action selection given the probabilities
        self.action = np.random.choice(indices, p=self.p.reshape(1, -1).squeeze()) 
        return self.action

    def observeReward(self, profit, scaling = 1): # loss is the difference between the maximum 
        # theoretical profit (if the price posted to the buyer is 1 and to the
        # seller is 0 and the trade gets accepted by both parties i.e. their
        # valuations are 1 and 0 respectively) and the profit from the actual trade
        loss = 1 / scaling - profit # scaling factor needed for accurate 
        # computations for EXP3
        self.w = self.w * np.exp(-self.learning_rate * loss) # weights of all
        # the actions are updated in the full feedback setting, loss is a vector
        epsilon = 1e-12  # small regularization constant preventing weight decay
        self.w = self.w + epsilon
        self.w = self.w / np.sum(self.w) # normalize weights for numerical stability
        

class EXP3(Hedge):
    def __init__(self, learning_rate, nActions):
        super().__init__(learning_rate, nActions)

    def nextAction(self, indices = None):
        return super().nextAction(indices)
    
    def observeReward(self, profit, indices = None): # profit - scalar
        if indices is None:
            action_idx = self.action
        else: # for importance weighting, we need the probability of the selected
        # action given by the vector of restricted distribution so the probability
        # according to which the action was selected in this round
            action_idx = np.where(indices == self.action)[0]
        p_action = self.p[action_idx] # the actual probability according to which an
        # action was drawn
        profits = np.zeros(self.nActions).reshape(-1,1) 
        # importance weighting to make the estimate unbiased, one-bit feedback
        profits[self.action] = profit / p_action 
        super().observeReward(profits, scaling = p_action)
        
    
        
        
        
        
        
        