import numpy as np

np.random.seed(123)



def stochastic_valuations(a_b, b_b, a_s, b_s):
    """
    We reasonably assume that the seller's valuations are slightly right-skewed
    and the buyer's slightly left-skewed and both in the interval [0, 1].
    """
    return np.random.beta(a=a_b, b=b_b), np.random.beta(a=a_s, b=b_s)

