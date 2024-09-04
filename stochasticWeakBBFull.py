
import numpy as np
from stochastic_val import stochastic_valuations
from profit import profit
from regret import regret
import matplotlib.pyplot as plt

np.random.seed(12)



# standard UCB - functional for full feedback, stochastic setting and weak BB

class UCB:
    def __init__(self, nActions):
        self.nActions = nActions
        self.Q = np.zeros(nActions)  
        self.rewards = np.zeros(nActions) 
        self.N = 0 # in full feedback we get information for all arms so as if all of them were played 
        # self.t = 0  
        self.action = None
        self.acc_profit = []  

    def nextAction(self):
        # self.t += 1
        ucb_values = self.Q # + np.sqrt((2 * np.log(self.t)) / (self.N + 1e-5)) 
        # Is it not enough to explore every action at least once?
        self.action = np.argmax(ucb_values)
        return self.action

    def observeReward(self, profits):
# profit vector
        self.rewards+= profits
        self.N+= 1
        self.Q= self.rewards/self.N
        self.acc_profit.append(profits[self.action] + (self.acc_profit[-1] if self.acc_profit else 0.0))


    def reset(self): # things which have change get back to the original state
        self.Q = np.zeros(self.nActions)
        self.rewards = np.zeros(self.nActions)
        self.N = 0
        # self.t = 0
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
    
    # ADD A FUNCTION WITH THIS OPERATION VECTORIZED
    potential_profits = np.array([p[0] - p[1] for p in price_grid])
    curr_profits = potential_profits * np.array([p[0]<=b_t for p in price_grid]) * np.array([p[1]>=s_t for p in price_grid])
    
    algo.observeReward(curr_profits)
    
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
fig.suptitle('Stochastic Setting, Global Budget Balance, Bandit Feedback', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
plt.show()