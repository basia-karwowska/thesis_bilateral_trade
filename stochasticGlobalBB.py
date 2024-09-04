import numpy as np
from stochastic_val import stochastic_valuations
from profit import profit
from regret import regret
import matplotlib.pyplot as plt

np.random.seed(10)

# Only for bandit feedback
class UCB_globalBB:
    def __init__(self, p_b, p_s, initial_budget = 0, exploration_probability = 0.2, p_decay = 0.95, exploration_scaling = 1):
        
        self.budget = initial_budget
        self.p_b = p_b # buyer's prices, ndarray
        self.p_s = p_s # seller's prices
        self.n_b = len(self.p_b)
        self.n_s = len(self.p_s)
        
        self.price_grid = np.array([(p_b[i], p_s[j]) for i in range(self.n_b) for j in range(self.n_s) if p_b[i]>=p_s[j]]) 
        self.b = np.zeros(self.n_b) # attempt to make some use of side information; can be adapted
        # probabilities that the current valuations is a corresponding element of p_b?
        self.s = np.zeros(self.n_s)
        
        self.nActions = len(self.price_grid)
        self.Q = np.zeros(self.nActions)
        self.rewards = np.zeros(self.nActions)
        self.N = np.zeros(self.nActions)  
        self.t = 0  
        self.action = None # defined as price pairs
        self.action_idx = None
        self.p = exploration_probability
        self.exploration = False 
        self.c = exploration_scaling
        self.p_decay = p_decay
        self.acc_profit = []
        

    def nextAction(self):
        self.t += 1 
        if np.random.rand() < self.p: 
            # print("exploration move")
            # To get 2-bit feedback (but at some cost)
            self.exploration = True
            buyer_explore = np.random.choice([True, False])
            if buyer_explore: 
                allowed_idx = np.where(self.p_b >= 1 - self.budget)[0] 
                idx = np.random.choice(allowed_idx)
                self.action = (self.p_b[idx], 1)
            else:
                allowed_idx = np.where(self.p_s <= self.budget)[0]
                idx = np.random.choice(allowed_idx)
                self.action = (0, self.p_s[idx])
            # self.p /= np.log(self.t + 1) # logarithmic decay of exporation probability
            self.p *= self.p_decay
        else:
            self.exploration = False
            b_sum=np.sum(self.b)
            s_sum=np.sum(self.s)

            weighted_potential_reward = np.array([self.b[i]/max(b_sum, 1) * self.p_b[i] - self.s[j]/max(s_sum, 1) * self.p_s[j] for i in range(self.n_b) for j in range(self.n_s) if self.p_b[i]>=self.p_s[j]]) 
            # print(weighted_potential_reward)
            ucb_values_mod = self.c * weighted_potential_reward + self.Q + np.sqrt((2 * np.log(self.t)) / (self.N + 1e-5)) 
            # print(ucb_values_mod)
            # ucb_max = np.max(ucb_values_mod)
            self.action_idx = np.argmax(ucb_values_mod)
            self.action = self.price_grid[self.action_idx] 
        return self.action

    def observeReward(self, profit): 
        self.budget += profit
        p_b, p_s = self.action
        if self.exploration is True: 
            if p_s==1:
                if profit < 0: 
                    self.b[self.b >= p_b] += 1
                else: 
                    self.b[self.b<p_b] += 1
            else:
                if profit < 0: 
                    self.s[self.s<=p_s] += 1
                else: 
                    self.s[self.s>p_s] += 1
        else: 
             self.rewards[self.action_idx]+= profit
             self.N[self.action_idx]+= 1
             self.Q[self.action_idx]= self.rewards[self.action_idx]/self.N[self.action_idx]
        self.acc_profit.append(profit + (self.acc_profit[-1] if self.acc_profit else 0.0))
    
            

    def reset(self):
        # self.budget = initial_budget
        self.b = np.zeros(self.n_b)
        self.s = np.zeros(self.n_s)
        self.Q = np.zeros(self.nActions)
        self.rewards = np.zeros(self.nActions)
        self.N = np.zeros(self.nActions)
        self.t = 0
        self.action = None
        self.action_idx = None
        self.exploration = False 
        self.acc_profit = []

    def results(self):
        return self.acc_profit
    
    
    
    
T = 10000 # adjust to more
discret = 100
prices_b = np.linspace(0, 1, discret)
prices_s = np.linspace(0, 1, discret)
algo = UCB_globalBB(prices_b, prices_s) 

profits = [] # can be recreated from algo.acc_profits
regrets = []
valuations = []


for t in range(T):
    b_t, s_t = stochastic_valuations(5, 2, 2, 5) 
    # print(b_t, s_t)
    valuations.append((b_t, s_t))
    p_b_t, p_s_t = algo.nextAction() 
    # print(p_b_t, p_s_t)
    
    curr_profit = profit(p_b_t, p_s_t, b_t, s_t) 
    # print(curr_profit)
    profits.append(curr_profit)
    algo.observeReward(curr_profit)
    
    curr_regret = regret(valuations, algo.price_grid, profits) 
    regrets.append(curr_regret) 

profits = np.array(profits)
regrets = np.array(regrets)

acc_profits = np.array(algo.results())
loss_idx = np.where(np.array(profits)<0)[0]
# print(loss_idx)
losses = np.zeros(len(profits))
if loss_idx.size > 0:
    # print(loss_idx)
    losses[loss_idx] = profits[loss_idx]
    
    
# Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# First plot: Regret Over Time
axs[0, 0].plot(regrets, label='Regret')
axs[0, 0].set_xlabel('Round')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].set_title('Regret Over Time')

# Second plot: Accumulated Loss Over Time
axs[0, 1].plot(losses, label='Losses - explorative actions')
axs[0, 1].set_xlabel('Round')
axs[0, 1].set_ylabel('Value')
axs[0, 1].legend()
axs[0, 1].set_title('Loss from Exploration Over Time')

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