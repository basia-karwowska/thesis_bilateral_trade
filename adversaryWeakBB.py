'''
Adversarial setting, weak budget balance
'''
# add seed

from adversary import adversarial_rewards, adversarial_strategy1, adversarial_strategy2
from hedge import Hedge, EXP3
from regret import regret_adv
import numpy as np
import matplotlib.pyplot as plt

# Leveraged vectorized operations

# TO BE SELECTED by the user
T = 10000 # number of time steps, rounds
discret = 100 # discretization steps
learning_rate = 0.1 # what is a common choice?
regret_minimizer = Hedge # Hedge for full-feedback and EXP3 for partial feedback
feedback = "Full" # consistent with the choice of regret minimizer -
# either (EXP3, Bandit) or (Hedge, Full); put either: "Bandit" or "Full"
adversarial_strategy = adversarial_strategy2

# PRICES
# Price grid enforcing weak budget balance constraint
prices = np.linspace(0, 1, discret)
price_grid = [(p1, p2) for p1 in prices for p2 in prices if p1 >= p2]
# first price is for buyer and second for seller
price_grid = np.array(price_grid)

# ADVERSARY
# Reward vectors secretly selected by an adversary for all time steps T

K = len(price_grid) # number of arms, actions, price pairs, nActions
# we can use also another adversarial strategy
rewards = adversarial_rewards(price_grid, adversarial_strategy, T) 

# REGRET MINIMIZER initialization
algo = regret_minimizer(learning_rate, K)

# For regret computation
profits = [] 
regrets = []

# Learning

for t in range(T):
    
    # Pulling an arm and reading profit and regret; they are computed the same
    # way for both full and 1-bit feedback as profit depends only on the action
    # being chosen, arm being pull and not on the information about all other
    # arms. For regret computation, we use information about all other arms
    # anyways and compute it externally to regret minimizer, so the notion
    # of feedback is not relevant here
    
    action_idx = algo.nextAction() # arm is pulled, gets selected probabilistically
    profit = rewards[t, action_idx] # reward corresponding to the arm pulled and
    # time step t is received (profit in our case), read from an appropriate
    # tth element (vector) of the deterministic sequence
    profits.append(profit) # profit is stored for regret computation
    regret = regret_adv(rewards, profits) # regret in the current round computed
    regrets.append(regret) 

    # Computing losses to be passed to regret minimizer for learning; it differs
    # for Hedge and EXP3 (full feedback vs bandit feedback) as different amount
    # of information is available; for 1-bit feedback only the information about
    # the arm being pulled so only its weight gets updated and we use importance
    # weighting to account for this limited information, while for full feedback,
    # in loss computation we consider all the arms as we have information about
    # the rewards, losses associated with each arm, so we update all the weights,
    # from the perspective of loss determination, having this info is equivalent
    # to playing all these arms as for the loss computation just the knowledge
    # about the reward is needed (note that since in full feedback it is as if
    # an arm is pulled, we actually do not pull it so this does not impact
    # profit and regret, which are realized quantities, it just helps the algo
    # learn more quickly, update all the weights at each time step instead of just
    # 1 weight)
    if regret_minimizer == EXP3:
        losses = np.zeros(K) # initialize vector of losses to be observed by regret minimizer
    # loss = 1 - profit, so the difference between the maximum potential profit
    # and the actual profit (loss is 0 in the extreme case of selecting 1 price
    # for the buyer and 0 for the seller, while is 1 for the trade not happening
    # logic: prices offered but rejected - the worst case as we get the profit
    # at its lower bound so loss is the greatest, better to play a weak price
    # pair with small positive profit than a price pair which gets rejected
    # for one-bit feedback we do not know about the other actions so we
    # do not update their weights
        losses[action_idx] = 1-profit # the smaller the profit from an action,
    # the greater the loss and ofc loss is from 0 to 1, bounded just like profit
    
    elif regret_minimizer == Hedge:
        losses = (1 - rewards[t, :]).reshape(-1, 1) # we consider rewards for all actions
    # Update regret minimizer
    algo.observeLoss(losses)
        

# Results
acc_loss = algo.results()[0] 
acc_profits = np.cumsum(profits)

# Plot regrets, acc_loss, cum_profits, profits

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
fig.suptitle('Adversarial Setting, Weak Budget Balance, '  + feedback + ' Feedback', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
plt.show()



# Fitting linear function to regret - does it evolve at a sublinear rate?