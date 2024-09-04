from abstract_regret_minimizer import AbstractRegretMinimizer
import numpy as np

# Source: andCelli github

class Hedge(AbstractRegretMinimizer):
    def __init__(self, learning_rate, nActions):
        self.learning_rate = learning_rate
        self.nActions = nActions
        self.w = np.ones(nActions)[:,np.newaxis]
        self.p = np.ones(nActions) / nActions
        self.action = None

        self.acc_loss = []
        self.acc_cost = []

    def nextAction(self, indices = None, context=None):
        if indices is None:
            indices = np.arange(self.nActions)
            self.p = self.w / np.sum(self.w)
        else:
            self.p = self.w[indices] / np.sum(self.w[indices])   
        
        self.action = np.random.choice(indices, p=self.p.reshape(1,-1).squeeze())
        return self.action

    def observeLoss(self, loss, cost=None): # loss is the difference between 
        # the maximum potential profit (only theoretical if buyer's valuation
        # is 1 and seller's 0 and the profit from the actual trade, the wider
        # the gap the better and the smaller this difference and the smaller loss)
        
        # loss actually incurred is loss by the action played and this is part
        # of the accumulated loss
        
        # only loss of the action played is included in the accumulated loss
        # while losses for all potential actions (as diclosed through full
        # feedback setting get used to update the weights)
        # I changed loss[self.action][0] to loss[self.action]
        self.acc_loss.append(loss[self.action] + (self.acc_loss[-1] if self.acc_loss else 0.0))
        if cost is not None:
            self.acc_cost.append(cost[self.action][0] +  (self.acc_cost[-1] if self.acc_cost else 0.0))
        self.w = self.w * np.exp(-self.learning_rate * loss) # weights of all
        # the actions are updated in the full feedback setting

    def indexFilter(self, price_grid, budget):
        indices = np.where(price_grid[:, 1] - price_grid[:, 0] <= budget) # globally budget balanced price pairs
    
    def reset(self):
        self.action = None
        self.w = np.ones(self.nActions)[:,np.newaxis]
        self.p = np.ones(self.nActions) / self.nActions

        self.acc_loss = []
        self.acc_cost = []

    def results(self):
        return self.acc_loss, self.acc_cost, None



class EXP3(Hedge):
    def __init__(self, learning_rate, nActions):
        super().__init__(learning_rate, nActions)

    def nextAction(self, indices = None, context=None):
        return super().nextAction(context)
    
    def observeLoss(self, loss, cost=None):
        ell_hat = np.zeros(self.nActions).reshape(-1,1) 
        # importance weighting, one-bit feedback
        # importance weighting
        ell_hat[self.action] = loss[self.action] / self.p[self.action] 
        super().observeLoss(ell_hat)
        
        