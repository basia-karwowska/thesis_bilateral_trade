import numpy as np
from stochastic_val import stochastic_valuations
from profit import profit
from regret import regret
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from ucb_bandit import UCB



# standard UCB - functional for 1-bit feedback, stochastic setting and weak BB
# ucb with side information allows to incorporate the estimations of probabilities
# Bernouilli, based on variables (valuations) which themselves follow beta
# distribution, so they can be represented by CDF however parameters of beta
# distributions are not known; these probability estimations are estimations
# of Q using a different approach; Q estimated throughout the algo and these
# expected profits through moves violating the global budget balance
# Modified part is action selection, which also depends on expected profit
# estimated using other method with 2-bit feedback


np.random.seed(123)

T = 10000
discret = 20
prices_b = np.linspace(0, 1, discret) 
prices_s = np.linspace(0, 1, discret)
# Weakly budget balanced price grid (standard actions), passed to UCB
price_grid = np.array([(p1, p2) for p1 in prices_b for p2 in prices_s if p1 >= p2])

K = len(price_grid) 
c = 0.05
algo = UCB(K, c)

profits = []
regrets = []
valuations = [] 

# Additional parts - for global budget balanced exploratory moves
budget = 0 # initial budget, it must not go below 0 in any round
exploration_prob = 0.5 # initial probability of playing an exploratory move
# of the form (p_b, 1) or (0, p_s), violating the global budget balance
prob_decay = 0.95 # decay of probability of playing loss-incurring actions 

# Vectors with estimated beta distributions to model the uncertain probabilities
# of acceptance by the buyer and the seller for given buyer's and seller's
# prices, respectively. Initialization with uninformative priors, corresponding
# to uniform distributions.

# Estimated beta distribution parameters for buyer's prices
beta_params_b = np.array([[1, 1] for i in range(discret)])
# Estimated beta distribution parameters for seller's prices
beta_params_s = np.array([[1, 1] for i in range(discret)])

# Expected profits for price pairs from the price grid given the estimations
# of their acceptance probabilities (best guesses of these probabilities are
# expected values of the corresponding estimated beta distributions).

# Initialization of expected profits given uninformative priors - with
# expected values of 1/2 for each probability of acceptance.
expected_profits = np.array([(p[0]-p[1]) * 1/2 * 1/2 for p in price_grid])

for t in range(T):
    b_t, s_t = stochastic_valuations(5, 2, 2, 5) 
    valuations.append((b_t, s_t))
    
    # With probability equal to exploration_prob, play a loss-incurring action 
    if np.random.rand() < exploration_prob: 
        # There is always a possibility to post (0, 0) and (1, 1) respecting the
        # budget so the set of valid variable prices is not empty.
        
        # 4 distinct scenarios and distinct updates to estimated beta distributions
        # and to estimated profits: buyer's side explored & buyer accepts,
        # buyer's side explored & buyer rejects, seller's side explored & seller
        # accepts, seller's side explored & seller rejects
        
        exploration_prob *= prob_decay # as a loss-incurring action is played, the
        # probability of playing a loss-incurring action in the next rounds declines
        
        # Explore buyer's or seller's side with equal probability
        buyer_explore = np.random.choice([True, False])
        if buyer_explore:
            # Indices of buyer's prices respecting global budget balance
            valid_idx = np.where(prices_b>=1-budget)[0]
            
            # Indices (with respect to valid_idx array, which itself stores 
            # indices with respect to arrays prices_b and beta_params_b) of the 
            # less explored prices (the ones whose distributions were updated 
            # fewer times, whose sum of beta distribution parameters is the 
            # smallest) from the ones present in valid_idx array i.e. among the 
            # ones allowed given the global budget balance constraint:
            less_explored = np.where(np.sum(beta_params_b[valid_idx, :], axis=1)==
                                     np.min(np.sum(beta_params_b[valid_idx, :], axis=1)))[0]
            # Transforming indices with respect to valid_idx array to indices
            # with respect to prices_b array (original point of reference)
            indices = valid_idx[less_explored] 
            # Random choice of a price to explore from the ones allowed by
            # the global budget balance and from the ones less explored
            idx = np.random.choice(indices)
            p_b = prices_b[idx]
            curr_profit = profit(p_b, 1, b_t, s_t)
            if curr_profit != 0: # if trade happened, update alpha parameters
            # of distributions of p_b and lower prices (if trade accepted for 
            # p_b, it would have also been accepted for prices lower than p_b)
                beta_params_b[:idx+1, 0] += 1
            else: # trade did not happen as the price was too high for the 
            # buyer so also higher prices would not be accepted - update beta
            # parameters of distributions of prices greater or equal to p_b
                beta_params_b[idx:, 1] += 1   
        else: # explore seller's side - similar logic respecting global budget
        # balance and encouraging less explored actions, adjusted for the seller
            valid_idx = np.where(prices_s<=budget)[0] 
            less_explored = np.where(np.sum(beta_params_s[valid_idx, :], axis=1)
                                     ==np.min(np.sum(beta_params_s[valid_idx, :], axis=1)))[0]
            indices = valid_idx[less_explored]
            idx = np.random.choice(indices)
            p_s = prices_s[idx]
            curr_profit = profit(0, p_s, b_t, s_t)
            
            if curr_profit != 0: # if seller accepted
                beta_params_s[idx:, 0] += 1 # if a seller accepted p_s, they 
                # would have also accepted prices higher than p_s - update alphas
            else:
                beta_params_s[:idx+1, 1] += 1 # update betas
    
        # Expected profit: product of the potential profit of a price pair
        # and the estimated probabilities of acceptance of individual prices.
        # Estimated probabilities are expected values of corresponding beta
        # distributions used to model uncertainty of acceptance probabilities.
        # Mapping between prices and indices given discretization level:
        # p_i=i/(discret-1)  so i=p_i*(discret-1)
        expected_profits = np.array([(p[0]-p[1]) * (beta_params_b[int((discret-1)*p[0])][0]/
            (beta_params_b[int((discret-1)*p[0])][0] + beta_params_b[int((discret-1)*p[0])][1])) * 
            (beta_params_s[int((discret-1)*p[1])][0]/ (beta_params_s[int((discret-1)*p[1])][0] + 
                                  beta_params_s[int((discret-1)*p[1])][1])) for p in price_grid])
    
    else: # standard UCB action
        action_idx = algo.nextAction(expected_profits)
        p_b_t, p_s_t = price_grid[action_idx]
        curr_profit = profit(p_b_t, p_s_t, b_t, s_t) 
        algo.observeReward(curr_profit)
    
    budget += curr_profit
    profits.append(curr_profit)
    curr_regret = regret(valuations, price_grid, profits) 
    regrets.append(curr_regret) 
 
profits = np.array(profits)
acc_profits = np.cumsum(profits)
regrets = np.array(regrets)



np.savetxt("stoch_global_bandit_regrets.csv", regrets, delimiter=",", header="regret", comments='')
np.savetxt("stoch_global_bandit_profits.csv", acc_profits, delimiter=",", header="profit", comments='')

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
fig.suptitle('Stochastic Setting, Global Budget Balance, Bandit Feedback', fontsize=16)

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
