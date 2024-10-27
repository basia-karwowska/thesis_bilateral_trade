import numpy as np
'''
Regret at a particular round is the cumulative quantity, sum of optimal gains 
from trade if only one price was to be set throughout t rounds up until now minus 
the actual sum of gains from trade.
Prices - discretized price grid.
'''


def regret(valuations, prices, act_profits): # vectorized implementation
    cum_profit = np.sum(act_profits) # actual profits in bilateral trade
    # Valuation history until time t with respect to which we aim to compute the regret
    valuations = np.array(valuations) 
    b = valuations[:, 0]
    s = valuations[:, 1]
    # Optimal price pair computation
    prices = np.array(prices) # all possible price pairs
    p_b = prices[:, 0]
    p_s = prices[:, 1]
    pot_profits = p_b - p_s # potential profits
    b_acc = p_b[:, np.newaxis] <= b # boolean matrix of buyer's acceptances of
    # the prices in the past rounds if their valuation was b in all of these rounds
    s_acc = p_s[:, np.newaxis] >= s # the same for the seller
    # A vector of total profits accumulated throughout t rounds for all K price 
    # pairs if they were played consistently throughout all t rounds
    profits = np.sum(b_acc * s_acc * pot_profits[:, np.newaxis], axis=1)  
    idx_max = np.argmax(profits) # index of the profit-maximizing price pair with hindsight
    profit_max = profits[idx_max] # maximum cumulative profit in hindsight
    # cum_profit is deterministic so does not impact the maximization
    return profit_max - cum_profit

'''
def cum_regrets(valuations, prices, cum_profits):
    # Input: tuple of T valuations and T cumulative profits
    
   # cum_profit = np.sum(act_profits) # actual profits in bilateral trade
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
    pot_profits = p_b - p_s # potential profits for each price pair so length len(prices)
    # n = len(price_diff)
    
    # len(prices) x T matrix with decisions regarding price posted (row)
    # at a given round T (so conditional on valuation at time T) (column)
    b_act = p_b[:, np.newaxis] <= b # buyer accepts - boolean matrix
    # similar for the seller's matrix
    s_act = p_s[:, np.newaxis] >= s
    # print(b_act, s_act)
    # total profit accumulated throughout t rounds for all n price pairs if 
    # a single pair was played throughout all t rounds
    # T by len(prices) matrix
    # print(b_act * s_act * pot_profits[:, np.newaxis])
    # Element-wise product of matrices with buyer's and seller's decision
    # and each row of the matrix is multiplied by the potential profit to
    # get realized profits at each round for each price pair
    # We take row sums of this matrix, so for each price pair (profits that
    # could be realized if we post this price pair over T round), of T elements
    trades = b_act * s_act * pot_profits[:, np.newaxis]
    profits = np.sum(trades, axis=1)  
    # print(profits)
    id_max = np.argmax(profits) # index of the profit-maximizing price pair with hindishgt
    
    # OPTIMAL PRICE PAIR IN HINDSIGHT - profit maximizing
    p_opt = prices[id_max]
    # We take a row of this price pair and compute the cumsum array to get cumulative
    # profits if this price pair was played over T rounds
    profits_opt = trades[id_max, :]
    # print(profits_opt)
    cum_profits_opt = np.cumsum(profits_opt)
    
    
    # profit_max = profits[id_max] # maximum cumulative profit in hindsight
    # cum_profit is deterministic so does not impact this
    # return profit_max - cum_profit
    return cum_profits_opt - cum_profits
'''

def regret_adv(rewards, act_profits): # input: T by K rewards matrix & profit vector
    t = len(act_profits) # how many rounds have passed
    cum_profit = np.sum(act_profits)
    curr_arms_rewards = np.sum(rewards[:t, :], axis=0) # column sums - rewards
    # of each arm (fixed price pair strategy) up until round t
    # Maximum over column sums - sums of rewards over T rounds for given arms
    return np.max(curr_arms_rewards) - cum_profit
