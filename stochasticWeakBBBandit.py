import numpy as np
from stochastic_val import stochastic_valuations
from profit import profit
from regret import regret
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(11)



# standard UCB - functional for 1-bit feedback, stochastic setting and weak BB

class UCB:
    def __init__(self, nActions, c=1):
        self.nActions = nActions # number of arms, actions, price pairs
        self.Q = np.zeros(nActions) # average rewards  (here: profits)
        self.X = np.zeros(nActions) # cumulative rewards (here: profits)
        self.N = np.zeros(nActions) # number of arm pulls
        self.t = 0  
        self.action = None
        self.c = c # scaling factor for the exploration term
        # self.acc_profit = []

    def nextAction(self):
        # Action selection
        self.t += 1
        # Initial exploration: ensure that each action is selected at least once
        if np.min(self.N) == 0:
            self.action = np.argmin(self.N)
        # Selection of an price pair maximizing the UCB value
        else:
            ucb_values = self.Q + self.c * np.sqrt((2 * np.log(self.t)) / self.N) 
            self.action = np.argmax(ucb_values)
        return self.action

    def observeReward(self, profit):
        # Parameters updates
        self.X[self.action] += profit # update the reward vector
        self.N[self.action] += 1 # update the vector of action counts
        # Update the average reward vector
        self.Q[self.action] = self.X[self.action] / self.N[self.action]
        # self.acc_profit.append(profit + (self.acc_profit[-1] if self.acc_profit else 0.0))

    def reset(self):
        self.Q = np.zeros(self.nActions)
        self.X = np.zeros(self.nActions) 
        self.N = np.zeros(self.nActions)
        self.t = 0
        self.action = None
        # self.acc_profit = []

    def results(self):
        return self.acc_profit
        
        
         
        
    
'''
OPTION TO CONSIDER
In this case, the key challenge is to model the uncertainty over whether a 
trade happens (binary outcome), which depends on the buyer and seller valuations, 
and the corresponding reward is deterministic once the trade happens.
'''

class BayesUCBProfit:
    def __init__(self, nActions, mu_prior=0, sigma_prior=1, known_variance=1):
        self.nActions = nActions
        # Initialize posterior parameters for each action (mean and variance for Gaussian)
        self.mu = np.ones(nActions) * mu_prior  # Prior mean (expected profit)
        self.sigma = np.ones(nActions) * sigma_prior  # Prior standard deviation
        self.N = np.zeros(nActions)  # Number of times each arm was pulled
        self.known_variance = known_variance  # Known variance of the reward distribution
        self.t = 0  # Time step
        self.action = None
        # self.acc_profit = []  # To store accumulated profit

    def nextAction(self):
        self.t += 1
        quantiles = np.zeros(self.nActions)
        
        # Compute 1 - 1/t quantile for the reward distribution of each arm
        for i in range(self.nActions):
            # Compute the posterior variance for the mean of the reward distribution
            posterior_sigma = self.sigma[i] / np.sqrt(1 + self.N[i])
            # Compute the 1 - 1/t quantile from the Gaussian posterior
            quantiles[i] = stats.norm.ppf(1 - 1 / self.t, loc=self.mu[i], scale=posterior_sigma)
        
        # Select the arm with the highest quantile (upper confidence bound for profit)
        self.action = np.argmax(quantiles)
        return self.action

    def observeReward(self, reward):
        # Update the number of times this action was selected
        self.N[self.action] += 1
        
        # Compute the posterior update for the Gaussian mean (reward)
        # Using Bayesian updating for a normal likelihood with known variance
        self.mu[self.action] = (self.mu[self.action] * (self.N[self.action] - 1) + reward) / self.N[self.action]
        self.sigma[self.action] = np.sqrt((self.known_variance**2) / self.N[self.action])
        
        # Update the accumulated profit
        # self.acc_profit.append(reward + (self.acc_profit[-1] if self.acc_profit else 0.0))

    def reset(self):
        # Reset all parameters to the initial state
        self.mu = np.ones(self.nActions)
        self.sigma = np.ones(self.nActions)
        self.N = np.zeros(self.nActions)
        self.t = 0
        self.action = None
        # self.acc_profit = []

    # def results(self):
        # return self.acc_profit

    

    
    
    


T = 10000 # time horizon
discret = 20 # discretization level of the price grid
prices = np.linspace(0, 1, discret)
# Price grid guaranteeing weak budget balance - price pairs with nonnegative profit
price_grid = np.array([(p1, p2) for p1 in prices for p2 in prices if p1>=p2])
K = len(price_grid) # number of actions, arms, price pairs
c = 0.1 # scaling factor for the exploration term
algo = UCB(K, c) # regret minimizer

profits = [] # profits observed throughout all time steps
regrets = [] # stores cumulative regrets for T=t for t=1....T
valuations = [] # to compute the regret later for algo evaluation


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
    
    # Cumulative regret computation at T=t (if t was the final round)
    curr_regret = regret(valuations, price_grid, profits) 
    regrets.append(curr_regret) 
 
profits = np.array(profits)
acc_profits = np.cumsum(profits) 
regrets = np.array(regrets)


# regrets = cum_regrets(valuations, price_grid, acc_profits)





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

## Plot the original regret, linear fit, t^(1/2) fit and t^(2/3) fit

axs[0].plot(rounds, regrets, label='Regret', color='blue')  # Original regret curve
axs[0].plot(rounds, regret_linear_fit, label='Linear Fit', linestyle='--', color='red')  # Linear fit
axs[0].plot(rounds, regret_sublinear_23_fit, label=r'Sublinear Fit $t^{2/3}$', linestyle='--', color='green')  # t^(2/3) fit
axs[0].plot(rounds, regret_sublinear_12_fit, label=r'Sublinear Fit $t^{1/2}$', linestyle='--', color='yellow')  # t^(1/2) fit
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


# Print out the results
print(f"Linear Fit: MSE = {mse_linear:.3f}, RMSE = {rmse_linear:.3f}, R^2 = {r2_linear:.3f}")
print(f"Sublinear (t^(2/3)) Fit: MSE = {mse_sublinear1:.3f}, RMSE = {rmse_sublinear1:.3f}, R^2 = {r2_sublinear1:.3f}")
print(f"Sublinear (t^(1/2)) Fit: MSE = {mse_sublinear2:.3f}, RMSE = {rmse_sublinear2:.3f}, R^2 = {r2_sublinear2:.3f}")



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
