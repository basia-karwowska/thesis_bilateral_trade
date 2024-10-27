import numpy as np
from stochastic_val import stochastic_valuations
from profit import profit
from regret import regret
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from ucb_bandit import UCB


np.random.seed(123) # to make simulations reproducible

T = 10000 # time horizon
discret = 20 # discretization level of the price grid
prices = np.linspace(0, 1, discret)
# Price grid guaranteeing weak budget balance - price pairs with nonnegative profit
price_grid = np.array([(p1, p2) for p1 in prices for p2 in prices if p1>=p2])
K = len(price_grid) # number of actions, arms, price pairs
c = 0.03 # scaling factor for the exploration term
algo = UCB(K, c) # regret minimizer

profits = [] # profits observed throughout all time steps
regrets = [] # stores cumulative regrets for T=t for t=1....T
valuations = [] # to compute the regret later for the performance evaluation


for t in range(T):
    # Valuations generation
    b_t, s_t = stochastic_valuations(5, 2, 2, 5) 
    valuations.append((b_t, s_t))
    
    # Posting of price pairs
    action_idx = algo.nextAction()
    p_b_t, p_s_t = price_grid[action_idx]
    
    # Reward observed (and bandit feedback - whether or not the trade occured)
    curr_profit = profit(p_b_t, p_s_t, b_t, s_t) 
    profits.append(curr_profit)
    algo.observeReward(curr_profit)
    
    # Cumulative regret computation at T=t (as if t was the final round)
    curr_regret = regret(valuations, price_grid, profits) 
    regrets.append(curr_regret) 
 
profits = np.array(profits)
acc_profits = np.cumsum(profits) 
regrets = np.array(regrets)

np.savetxt("stoch_weak_bandit_regrets.csv", regrets, delimiter=",", header="regret", comments='')
np.savetxt("stoch_weak_bandit_profits.csv", acc_profits, delimiter=",", header="profit", comments='')


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

## Plot the original regret, linear fit, t^(1/2) fit, t^(2/3) fit and log(t) fit

axs[0].plot(rounds, regrets, label='Regret', color='blue')  # Original regret curve
axs[0].plot(rounds, regret_linear_fit, label='Linear Fit', linestyle='--', color='red')  # Linear fit
axs[0].plot(rounds, regret_sublinear_23_fit, label=r'Sublinear Fit $t^{2/3}$', linestyle='--', color='green')  # t^(2/3) fit
axs[0].plot(rounds, regret_sublinear_12_fit, label=r'Sublinear Fit $t^{1/2}$', linestyle='--', color='yellow')  # t^(1/2) fit
axs[0].plot(rounds, regret_sublinear_log_fit, label=r'Sublinear Log Fit $log(t)$', linestyle='--', color='orange')  # log fit
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

'''
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
'''


# Set the main title for the figure
fig.suptitle('Stochastic Setting, Weak Budget Balance, Bandit Feedback', fontsize=16)

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


