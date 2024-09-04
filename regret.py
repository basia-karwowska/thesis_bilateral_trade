import numpy as np
'''
Regret at a particular round is the cumulative quantity, sum of optimal gains f
rom trade if only one price was to be set throughout t rounds up until now minus 
the actual sum of gains from trade.
So of course regret grows from one round to another given it is a cumulative
quantity? But we want this growth to happen at a sublinear rate.
Prices - discretized price grid.
Call it regret for revenue
'''
def regret(valuations, prices, act_profits):
    cum_profit = np.sum(act_profits) # actual profits in bilateral trade
    # Valuations: valuation history until time t with respect to which we aim to compute the regret
    valuations = np.array(valuations) # change in the code to numpy array so you do not need to do it here
    # fix these numpy arrays later to use them consistenly instead of lists to be able to vectorize
    b = valuations[:, 0]
    s = valuations[:, 1]
    # Optimal price pair computation
    # t = len(valuations) # number of rounds so far
    prices = np.array(prices) # all possible price pairs
    p_b = prices[:, 0]
    p_s = prices[:, 1]
    pot_profits = p_b - p_s # potential profits
    # n = len(price_diff)
    b_acc = p_b[:, np.newaxis] <= b # buyer accepts - boolean matrix
    s_acc = p_s[:, np.newaxis] >= s
    # total profit accumulated throughout t rounds for all n price pairs if 
    # a single pair was played throughout all t rounds
    profits = np.sum(b_acc * s_acc * pot_profits[:, np.newaxis], axis=1)  
    id_max = np.argmax(profits) # index of the profit-maximizing price pair with hindishgt
    profit_max = profits[id_max] # maximum cumulative profit in hindsight
    # cum_profit is deterministic so does not impact this
    return profit_max - cum_profit




def regret_adv(rewards, act_profits): # T by K rewards matrix
    # maximum over column sums - sum of rewards over T rounds for given arm
    t = len(act_profits) # how many rounds have passed
    cum_profit = np.sum(act_profits)
    curr_arms_rewards = np.sum(rewards[:t, :], axis=0) # column sums - rewards
    # of each arm (fixed price pair strategy) up until round t
    return np.max(curr_arms_rewards) - cum_profit