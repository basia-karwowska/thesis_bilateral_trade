import numpy as np
from stochastic_val import stochastic_valuations
from profit import profit
from regret import regret
import matplotlib.pyplot as plt

np.random.seed(11)



# standard UCB - functional for 1-bit feedback, stochastic setting and weak BB
class UCB:
    def __init__(self, nActions):
        self.nActions = nActions
        self.Q = np.zeros(nActions)  
        self.rewards = np.zeros(nActions) 
        self.N = np.zeros(nActions)  
        self.t = 0  
        self.action = None
        self.acc_profit = []  

    def nextAction(self):
        self.t += 1
        ucb_values = self.Q + np.sqrt((2 * np.log(self.t)) / (self.N + 1e-5)) 
        # Is it not enough to explore every action at least once?
        self.action = np.argmax(ucb_values)
        
        '''
        if self.t <= self.nActions:
            # Explore each action at least once, otherwise we would get stuck
            # given our selection rule. To have some balance to start from.
            self.action = self.t - 1
        else:
            # Calculate the UCB values
            ucb_values = self.Q + np.sqrt((2 * np.log(self.t)) / (self.N + 1e-5))  
            # For numerical stability.
            # Added epsilon to avoid division by zero - but I reckon it's not an 
            # issue if we first make sure we play all the actions once before
            # using this rule of selecting an action (the if condition above)
            # The first term is exploitation - the greater the average reward associated
            # with an action, the greater ucb. The second term is about exploitation
            # the fewer times an action was played, the greater ucb, which encourages
            # trying less often selected actions more times.
            self.action = np.argmax(ucb_values) # select an action with the greatest ucb value
        '''
        return self.action

    def observeReward(self, profit):
        self.rewards[self.action]+= profit
        self.N[self.action]+= 1
        self.Q[self.action]= self.rewards[self.action]/self.N[self.action]
        self.acc_profit.append(profit + (self.acc_profit[-1] if self.acc_profit else 0.0))


    def reset(self):
        self.Q = np.zeros(self.nActions)
        self.N = np.zeros(self.nActions)
        self.t = 0
        self.action = None
        self.acc_profit = []

    def results(self):
        return self.acc_profit
    
    
    
    
    
    

    
    
    





T = 10000 # adjust to more
discret = 100
prices = np.linspace(0, 1, discret)
price_grid = np.array([(p1, p2) for p1 in prices for p2 in prices if p1 >= p2])

algo = UCB(len(price_grid))

profits = []
regrets = []
valuations = [] # to compute the regret later for algo evaluation

 
 
for t in range(T):
    b_t, s_t = stochastic_valuations(5, 2, 2, 5) 
    valuations.append((b_t, s_t))
    action_idx = algo.nextAction()
    p_b_t, p_s_t = price_grid[action_idx]
    
    curr_profit = profit(p_b_t, p_s_t, b_t, s_t) 
    profits.append(curr_profit)
    algo.observeReward(curr_profit)
    
    curr_regret = regret(valuations, price_grid, profits) 
    regrets.append(curr_regret) 
 
profits = np.array(profits)
regrets = np.array(regrets)
acc_profits = np.array(algo.results())


# Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# First plot: Regret Over Time
axs[0, 0].plot(regrets, label='Regret')
axs[0, 0].set_xlabel('Round')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].set_title('Regret Over Time')

# Second plot: Accumulated Loss Over Time
# axs[0, 1].plot(losses, label='Losses - explorative actions')
# axs[0, 1].set_xlabel('Round')
# axs[0, 1].set_ylabel('Value')
# axs[0, 1].legend()
# axs[0, 1].set_title('Loss from Exploration Over Time')

# Third plot: Accumulated Profit Over Time
axs[1, 0].plot(acc_profits, label='Cumulative Profit')
axs[1, 0].set_xlabel('Round')
axs[1, 0].set_ylabel('Value')
axs[1, 0].legend()
axs[1, 0].set_title('Accumulated Profit Over Time')


# Fourth plot: Profit Over Time
axs[1, 1].plot(profits, label='Profit')
axs[1, 1].set_xlabel('Round')
axs[1, 1].set_ylabel('Value')
axs[1, 1].legend()
axs[1, 1].set_title('Profit Over Time')


# Set the main title for the figure
fig.suptitle('Stochastic Setting, Weak Budget Balance, Bandit Feedback', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
plt.show()



