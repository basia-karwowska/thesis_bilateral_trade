import numpy as np

np.random.seed(123)

# Adversarial strategy at time t

def adversarial_strategy2(price_grid): # first buyer, second seller, n length of the price grid
    K=len(price_grid)
    valuations = np.zeros(price_grid.shape) # valuations initialization
    valuations[:, 0] = np.random.uniform(0, 1, K)
    valuations[:, 1] = np.random.uniform(0, 1, K)
    return valuations

    

def adversarial_strategy(price_grid):
    """
    Adversarial strategy where an adversary pre-defines the sequences of rewards
    for all arms. The adversary tries to make things difficult by setting the 
    seller's valuations higher than the seller's price and the buyer's valuation
    lower than the buyer's price with higher likelihood.
    """
    K = len(price_grid)
    
    # Generate perturbations for buyer and seller valuations
    perturbation_b = np.random.uniform(-0.4, 0.1, K)
    perturbation_s = np.random.uniform(-0.1, 0.4, K)
    
    # Apply perturbations
    valuations = price_grid.copy() # valuations initialization
    valuations[:, 0] += perturbation_b
    valuations[:, 1] += perturbation_s
    
    # Clamp the values to be within [0, 1] 
    valuations = np.clip(valuations, 0, 1)
    
    return valuations


def adversarial_rewards(price_grid, T): 
    """
    Adversarial sequence of rewards given the sequence of valuations.
    These rewards cannot be directly chosen by an adversary as profit
    depends on the price pairs chosen by the agent. An adversary can
    instead select whether the potential profit gets realized, so
    chooses the reward either 0 or equal to the price difference.

    """
    K = len(price_grid)
    rewards = np.zeros((T, K))
    for t in range(T):
        valuations = adversarial_strategy(price_grid)
        potential_profits = np.array([price_grid[i][0]-price_grid[i][1] for i in range(K)])
        trade_happened = (valuations[:, 0] > price_grid[:, 0]) & (valuations[:, 1] < price_grid[:, 1])
        rewards[t, :] = trade_happened * potential_profits
    return rewards
    

# ofc we have many possibilities to implement this function, also it may be 
# divese for buyer's and seller's maximum alpha and beta and also shift of
# the buyer and seller may occur separately not together
# we use beta for flexibility of representing different shapes and support
# between 0 and 1; maybe here we can allow just integers or also fractions?

def shifting_distributions_valuations(T, shift_prob = 0.1, alpha0_b = 1, beta0_b = 1, 
                                      alpha0_s = 1, beta0_s = 1, alpha_max = 7, beta_max = 7):
    valuations = []
    alpha_b = alpha0_b
    beta_b = beta0_b
    alpha_s = alpha0_s
    beta_s = beta0_s
    alphas = np.arange(1, alpha_max + 1)
    betas = np.arange(1, beta_max + 1)
    
    for t in range(T):
        if np.random.rand() < shift_prob: 
            # Random change of beta distributions' parameters
            alpha_b = np.random.choice(alphas)
            beta_b = np.random.choice(betas)
            alpha_s = np.random.choice(alphas)
            beta_s = np.random.choice(betas)
        b_t = np.random.beta(a = alpha_b, b = beta_b)
        s_t = np.random.beta(a = alpha_s, b = beta_s)
        valuations.append((b_t, s_t))
    return np.array(valuations)
            
            

    
