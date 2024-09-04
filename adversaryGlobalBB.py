# Add for example seed to both files and image with function of T fitted to show
# if regret evolves at a sublinear rate?
'''
Adversarial setting global budget balance
'''

from adversary import adversarial_rewards, adversarial_strategy1, adversarial_strategy2
from hedge import Hedge, EXP3
from regret import regret_adv
import numpy as np
import matplotlib.pyplot as plt

# TO BE FINE TUNED
budget = 0 # initialized budget; it gives how much loss we can at most incur
# at a particular round to respect global budget balance constraint, i.e. to
# never go below zero budget (at the beginning no budget was accumulated yet
# so at first iterations only nonnegative rewards are allowed; further positive
# rewards would build our budget and offer the regret minimizer more flexibility
# in proposing future prices - more flexibility may benefit learning)
# The rest are the same user-configurable parameters as for weak budget balance
T = 10000
discret = 100 # price grid is not limited now so will be larger for the
# same discretization level compared to the weak budget balance
learning_rate = 0.1 
regret_minimizer = Hedge # Hedge for full-feedback and EXP3 for partial feedback
feedback = "Full" # consistent with the choice of regret minimizer -
# either (EXP3, Bandit) or (Hedge, Full); put either: "Bandit" or "Full"
adversarial_strategy = adversarial_strategy2

# PRICE GRID - not enforcing any constraint as opposed to weak budget balance
# which was enforced at the level of the grid
prices = np.linspace(0, 1, discret)
price_grid = [(p1, p2) for p1 in prices for p2 in prices]
price_grid = np.array(price_grid)

# ADVERSARY - sequence of T reward vectors of length K (number of arms) 
# secretly selected by an adversary 

K = len(price_grid)
rewards = adversarial_rewards(price_grid, adversarial_strategy, T) 

# REGRET MINIMIZER initialization
algo = regret_minimizer(learning_rate, K)

# For regret computation
profits = [] 
regrets = []


# Learning

for t in range(T):
    valid_indices = np.where(price_grid[:, 1] - price_grid[:, 0] <= budget)[0]
    # print("1", t)
    # The learner posts prices respecting global budget balance, i.e. such
    # that p_s_t-p_b_t <= B_t (budget at round t)
    action_idx = algo.nextAction(indices = valid_indices)
    # print("2", action_idx)
    
    ## ONLY DIFFERENCE IS THIS PART WHICH ENSURES GLOBAL BUDGET BALANCE
    p_b, p_s = price_grid[action_idx]
    # print("3", p_b, p_s)
    # the weights get updated only if the action is played so calling next_action
    # just draws probabilistically new action index without impact on learning
    # The learner posts prices such that p_t-q_t <= B_t
    # while p_s - p_b > budget: # while loss incurred exceeds nonnegative budget,
    # which gives the maximum possible loss, we select another price pair
    # until we have p_s-p_b<=budget (which is always true if p_b>=p_s, so for
    # weak budget balance and true potentially for some pairs (p_b, p_s) whose
    # difference does not exceed the available budget)
        # print("4", p_b, p_s)
        # action_idx = algo.nextAction()
        # print("5", action_idx)
        # p_b, p_s = price_grid[action_idx]
        # print("6", p_b, p_s)
    # The price pair is in line with global budget balance
    profit = rewards[t, action_idx] # learner receives the reward; which is 
    # either 0 or p_b-p_s by construction function of prices, quantities known 
    # to the learner so fully disclosed to the learner
    # print("7", profit)
    # print("8", budget)
    budget += profit # update budget
    # print("9", budget)
    
    
    
    profits.append(profit) 
    regret = regret_adv(rewards, profits) 
    # print("10", regret)
    regrets.append(regret)
    
    # Feedback is revealed to the learner
    if regret_minimizer == EXP3:
        losses = np.zeros(K)
        losses[action_idx] = 1-profit
        # print("11", losses)
    elif regret_minimizer == Hedge:
        losses = (1 - rewards[t, :]).reshape(-1, 1) 
        # print("12", losses)
    algo.observeLoss(losses)
    
# Results
acc_loss = algo.results()[0] 
acc_profits = np.cumsum(profits)


# Plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# First plot: Regret Over Time
axs[0, 0].plot(regrets, label='Regret')
axs[0, 0].set_xlabel('Round')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].set_title('Regret Over Time')

# Second plot: Accumulated Loss Over Time
axs[0, 1].plot(acc_loss, label='Cumulative Loss')
axs[0, 1].set_xlabel('Round')
axs[0, 1].set_ylabel('Value')
axs[0, 1].legend()
axs[0, 1].set_title('Accumulated Loss Over Time')

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
fig.suptitle('Adversarial Setting, Global Budget Balance, '  + feedback + ' Feedback', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
plt.show()
    