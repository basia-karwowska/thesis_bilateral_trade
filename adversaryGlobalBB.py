'''
Adversarial setting global budget balance
'''

from adversary import adversarial_rewards #, adversarial_strategy1, adversarial_strategy2
from hedge_exp3 import Hedge, EXP3
from regret import regret_adv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
'''Loss-incurring actions handled internally by hedge and exp3 which propose next 
actions in a probabilistic and not deterministic way (in contrast to UCB) with 
respect to current parameters (weights vs ucb values) so we do not need to explicitly 
force playing actions violating weak budget balance, the algorithm may generate 
such actions itself (although with low probability); we here we do not use 
information from loss-incurring actions to infer distribution and affect the 
choices of actions as for adversarial case we do not have an assumption of a 
regular distribution, instead the information from loss-incurring actions affects 
relative weights of all actions by construction due to normalization'''

np.random.seed(123) 

T = 10000
discret = 20 
prices = np.linspace(0, 1, discret)
price_grid = np.array([(p1, p2) for p1 in prices for p2 in prices]) 

K = len(price_grid)
learning_rate =  0.4
regret_minimizer = Hedge # EXP3 for partial feedback, Hedge for full-feedback
feedback = "Full" # consistent with the choice of regret minimizer -
# either (EXP3, Bandit) or (Hedge, Full); put either: "Bandit" or "Full"
algo = regret_minimizer(learning_rate, K)
budget = 0

# Sequence of adversarial rewards computed internally by an adversary based on 
# its internally generated valuations (adversarial strategy) and 
# potential profits (determined by the price grid). In practice, an adversary
# chooses rewards between p_b_t-p_s_t and 0 through its choice of valuations.
rewards = adversarial_rewards(price_grid, T)

profits = [] 
regrets = []

for t in range(T):
    # Enforcing global budget balance - price pairs with p_s - p_b <= budget
    valid_indices = np.where(price_grid[:, 1] - price_grid[:, 0] <= budget)[0]
    # The learner posts prices respecting global budget balance, i.e. such
    # that p_s_t-p_b_t <= B_t (budget at round t): action choice from valid_indices
    action_idx = algo.nextAction(indices = valid_indices)
    curr_profit = rewards[t, action_idx] # learner receives the reward; which is 
    # either 0 or p_b-p_s by construction
    budget += curr_profit 
    profits.append(curr_profit) 
    curr_regret = regret_adv(rewards, profits) 
    regrets.append(curr_regret)
    
    # Feedback is revealed to the learner - the amount depends on the setting chosen
    if regret_minimizer == EXP3:
        reward = curr_profit
        algo.observeReward(reward, valid_indices)
    elif regret_minimizer == Hedge:
        reward = rewards[t, :].reshape(-1, 1) 
        algo.observeReward(reward)
    
profits = np.array(profits)
acc_profits = np.cumsum(profits)
regrets = np.array(regrets)



'''
Computing losses to be passed to regret minimizer for learning; it differs
for Hedge and EXP3 (full feedback vs bandit feedback) as different amount
of information is available; for 1-bit feedback only the information about
the arm being pulled so only its weight gets updated and we use importance
weighting to account for this limited information, while for full feedback,
in loss computation we consider all the arms as we have information about
the rewards, losses associated with each arm, so we update all the weights,
from the perspective of loss determination, having this info is equivalent
to playing all these arms as for the loss computation just the knowledge
about the reward is needed (note that since in full feedback it is as if
an arm is pulled, we actually do not pull it so this does not impact
profit and regret, which are realized quantities, it just helps the algo
learn more quickly, update all the weights at each time step instead of just
1 weight). 
Loss = 1 - profit, so the difference between the maximum potential profit
and the actual profit (just like profit is bounded in the interval [-1, 1],
loss is bounded in the interval [0, 2]: 0 for ideal case of trade happening
for (1, 0), between 0 and 1 for trade happening and respecting weak budget
balance, 1 for trade not happening, between 1 and 2 for trade happening
and violating weak budget balance (worst case of 2 for prices (0, 1))
'''

np.savetxt("adv_global_full_regrets.csv", regrets, delimiter=",", header="regret", comments='')
np.savetxt("adv_global_full_profits.csv", acc_profits, delimiter=",", header="profit", comments='')

# Plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# First plot: Regret Over Time and fitted linear and sublinear functions

## Fit a linear Function (y = a * t + b)
rounds = np.arange(1, T+1) 
linear_model = LinearRegression()
linear_model.fit(rounds.reshape(-1, 1), regrets)  
regret_linear_fit = linear_model.predict(rounds.reshape(-1, 1)) 

## Fit a sublinear (t^(2/3)) function (y = a * t^(2/3) + b)
transformed_rounds1 = rounds ** (2/3)  # Transform rounds into t^(2/3)
sublinear_23_model = LinearRegression()
sublinear_23_model.fit(transformed_rounds1.reshape(-1, 1), regrets)  
regret_sublinear_23_fit = sublinear_23_model.predict(transformed_rounds1.reshape(-1, 1))

## Fit a sublinear (sqrt(t)) function (y = a * t^(1/2) + b)
transformed_rounds2 = rounds ** (1/2)  # Transform rounds into t^(2/3)
sublinear_12_model = LinearRegression()
sublinear_12_model.fit(transformed_rounds2.reshape(-1, 1), regrets)  
regret_sublinear_12_fit = sublinear_12_model.predict(transformed_rounds2.reshape(-1, 1))

# Fit a logarithmic function (y = a * log(t) + b)
transformed_rounds3 = np.log(rounds)  # Log-transform the rounds
sublinear_log_model = LinearRegression()
sublinear_log_model.fit(transformed_rounds3.reshape(-1, 1), regrets)  
regret_sublinear_log_fit = sublinear_log_model.predict(transformed_rounds3.reshape(-1, 1))

## Plot the original regret, linear fit, t^(1/2) fit, t^(2/3) fit and log fit

axs[0].plot(rounds, regrets, label='Regret', color='blue')  # Original regret curve
axs[0].plot(rounds, regret_linear_fit, label='Linear Fit', linestyle='--', color='red')  # Linear fit
axs[0].plot(rounds, regret_sublinear_23_fit, label=r'Sublinear Fit $t^{2/3}$', linestyle='--', color='green')  # t^(2/3) fit
axs[0].plot(rounds, regret_sublinear_12_fit, label=r'Sublinear Fit $t^{1/2}$', linestyle='--', color='yellow')  # t^(1/2) fit
axs[0].plot(rounds, regret_sublinear_log_fit, label=r'Sublinear Log Fit $log(t)$', linestyle='--', color='orange')  # log fit
# axs[0].plot(regrets, label='Regret')
axs[0].set_xlabel('Round')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].set_title('Regret Over Time with Linear and Sublinear Fits')

# Second plot: Accumulated Loss Over Time
axs[1].plot(acc_profits, label='Cumulative Profit')
axs[1].set_xlabel('Round')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].set_title('Accumulated Profit Over Time')


# Set the main title for the figure
fig.suptitle('Adversarial Setting, Global Budget Balance, ' + feedback + ' Feedback', fontsize=16)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
plt.show()



# Evaluation metrics for linear and sublinear fits
mse_linear = mean_squared_error(regrets, regret_linear_fit)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(regrets, regret_linear_fit)

mse_sublinear1 = mean_squared_error(regrets, regret_sublinear_23_fit)
rmse_sublinear1 = np.sqrt(mse_sublinear1)
r2_sublinear1 = r2_score(regrets, regret_sublinear_23_fit)

mse_sublinear2 = mean_squared_error(regrets, regret_sublinear_12_fit)
rmse_sublinear2 = np.sqrt(mse_sublinear2)
r2_sublinear2 = r2_score(regrets, regret_sublinear_12_fit)

mse_sublinear3 = mean_squared_error(regrets, regret_sublinear_log_fit)
rmse_sublinear3 = np.sqrt(mse_sublinear3)
r2_sublinear3 = r2_score(regrets, regret_sublinear_log_fit)


# Print out the results
print(f"Linear Fit: MSE = {mse_linear:.3f}, RMSE = {rmse_linear:.3f}, R^2 = {r2_linear:.3f}")
print(f"Sublinear (t^(2/3)) Fit: MSE = {mse_sublinear1:.3f}, RMSE = {rmse_sublinear1:.3f}, R^2 = {r2_sublinear1:.3f}")
print(f"Sublinear (t^(1/2)) Fit: MSE = {mse_sublinear2:.3f}, RMSE = {rmse_sublinear2:.3f}, R^2 = {r2_sublinear2:.3f}")
print(f"Sublinear (log(t))) Fit: MSE = {mse_sublinear3:.3f}, RMSE = {rmse_sublinear3:.3f}, R^2 = {r2_sublinear3:.3f}")






